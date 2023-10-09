import argparse
import os
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from types import SimpleNamespace
import math
import datetime
import pickle as pkl
import logging
import json
import random

from baselines.lstm import LSTMEncoderNaive, LatentLSTM
from baselines.patchtst import PatchtstNaive, LatentPatchtst
from baselines.mlp import MLPEncoderNaive, LatentMLP
from baselines.dlinear import DLinearNaive, LatentDLinear
from baselines.tcn import TCNNaive, LatentTCN
from utils import score_all, gen_split_time, init_logger
from dataloaders import BusLoadDataset, ECLDataset
from online_env import OnlineEnv, data_generator, OnlineEnvModel, OnlineMeta2Model, OnlineEnvTrain
from meta2net import MetaTS2Net


def fetch_train_cls(dataset, algorithm, mode, device, data_dir):
    if dataset == 'bus_1':
        # general parameters for bus dataset
        look_back_window = 96 * 3 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 87000) #总共时间长度, online_env parameters
        warm_up_interval = (0, 10000) # warm up 范围
        online_step_size = 7 * 96 # online过程中每个dataset的长度
        # 输入dim
        enc_in = 1
        # 输出dim
        
        enc_out = 1
        # meta_train长度
        meta_train_length = 80 
        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = BusLoadDataset(os.path.join(data_dir, 'bus_1.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='target', mode='S', warm_up_time=warm_up_interval)
        test = BusLoadDataset(os.path.join(data_dir, 'bus_1.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='target', mode='S', warm_up_time=warm_up_interval)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 10 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2
            
            
            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers

            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model


        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 8
            sim_loss_use_alpha = True
            sim_loss_score_num = 10


            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 256 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 16 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))
        










    elif dataset == 'bus_2':
        # general parameters for bus dataset
        look_back_window = 96 * 3 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 90786) #总共时间长度, online_env parameters
        warm_up_interval = (0, 10000) # warm up 范围
        online_step_size = 7 * 96 # online过程中每个dataset的长度
        # 输入dim
        enc_in = 1
        # 输出dim
        enc_out = 1
        # meta_train长度
        meta_train_length = 80 
        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = BusLoadDataset(os.path.join(data_dir, 'bus_2.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='load', mode='S', warm_up_time=warm_up_interval)
        test = BusLoadDataset(os.path.join(data_dir, 'bus_2.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='load', mode='S', warm_up_time=warm_up_interval)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 10 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2
            
            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1
            



            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 8
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
    
        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 40 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 10 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.3
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 8
            sim_loss_use_alpha = True
            sim_loss_score_num = 10


            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 6
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers

                

            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 16 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 8
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))








    elif dataset == 'bus_3':
        # general parameters for bus dataset
        look_back_window = 96 * 3 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 90818) #总共时间长度, online_env parameters
        warm_up_interval = (0, 10000) # warm up 范围
        online_step_size = 7 * 96 # online过程中每个dataset的长度
        # 输入dim
        enc_in = 1
        # 输出dim
        enc_out = 1
        # meta_train长度
        meta_train_length = 80 
        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = BusLoadDataset(os.path.join(data_dir, 'bus_3.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='load', mode='S', warm_up_time=warm_up_interval)
        test = BusLoadDataset(os.path.join(data_dir, 'bus_3.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='load', mode='S', warm_up_time=warm_up_interval)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 10 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2
            
            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1


            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 1
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
    
        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 30 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 6
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10


            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2


            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.1
            latent_wd = 0.2
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers
                

            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.1
            latent_wd = 0.1
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))









    # ETTH2    
    elif dataset == 'etth2':
        # general parameters for bus dataset
        look_back_window = 96 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 17500) #总共时间长度, online_env parameters
        warm_up_interval = (0, 2000) # warm up 范围
        online_step_size = 3 * 96 # online过程中每个dataset的长度

        # 输入dim
        enc_in = 7
        # 输出dim
        enc_out = 7
        # meta_train长度
        meta_train_length = 40 

        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = BusLoadDataset(os.path.join(data_dir, 'ETTh2.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='OT', mode='M', dt_col='date', warm_up_time=warm_up_interval)
        test = BusLoadDataset(os.path.join(data_dir, 'ETTh2.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='OT', mode='M', dt_col='date', warm_up_time=warm_up_interval)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim

            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.3

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # mesta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 1
            reg_latent_coef = 0
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 10 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = 512
            meta_loss_dim = 512
            latent_meta_loss_dim = 256
            latent_meta_loss_layers = 3
            param_generator_dim = 1024
            extrapolation_dim = 256
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = 192
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
          
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers

                

            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 32,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 40 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 8
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 32,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 8
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))







    # ETTm1    
    elif dataset == 'ettm1':
        # general parameters for bus dataset
        look_back_window = 96 * 3 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 69681) #总共时间长度, online_env parameters
        warm_up_interval = (0, 10000) # warm up 范围
        online_step_size = 7 * 96 # online过程中每个dataset的长度

        # 输入dim
        enc_in = 7
        # 输出dim
        enc_out = 7
        # meta_train长度
        meta_train_length = 70

        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = BusLoadDataset(os.path.join(data_dir, 'ETTm1.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='OT', mode='M', dt_col='date', warm_up_time=warm_up_interval)
        test = BusLoadDataset(os.path.join(data_dir, 'ETTm1.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='OT', mode='M', dt_col='date', warm_up_time=warm_up_interval)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim

            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.3

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 16 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # mesta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 6
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0.2
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers

                

            }
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 4,
                'd_ff': 128,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 30 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize 
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.1
            reg_latent_coef = 0.2
            latent_wd = 0
            sim_loss_dim = 512
            meta_loss_dim = 512
            latent_meta_loss_dim = 256
            latent_meta_loss_layers = 2
            param_generator_dim = 1024
            extrapolation_dim = 512
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0
            extrapolation_linear_dim = 512
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 4,
                'd_ff': 128,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 256 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 16 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 8
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'reg_latent_coef': reg_latent_coef,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))










    # ECL   
    elif dataset == 'ECL':
        # general parameters for bus dataset
        look_back_window = 96 # input_len
        forecast_horizon = 24 # ouptut_len
        total_time_interval = (0, 26304) #总共时间长度, online_env parameters
        warm_up_interval = (0, 5000) # warm up 范围
        online_step_size = 3 * 96 # online过程中每个dataset的长度

        # 输入dim
        enc_in = 12
        # 输出dim
        enc_out = 12
        # meta_train长度
        meta_train_length = 50 

        # 得到时间划分
        split_times = gen_split_time(total_time_interval, warm_up_interval, online_step_size, mode=mode)
        train = ECLDataset(os.path.join(data_dir, 'ECL.csv'), look_back_window, device, pred_len=forecast_horizon, target_col='OT', mode='M', dt_col='date', warm_up_time=warm_up_interval, col_num=12)
        test = deepcopy(train)
        
        if algorithm == 'naive_lstm':
            # baseline LSTM算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            reg_latent_coef = 0.1
            latent_wd = 0.2

            meta_train_epochs = 0  # meta train循环训练多少次
            warm_up_mode = None
            # 用于打印
            model_config = {
                'enc_in': enc_in, # lstm 输入 dim
                'enc_out': enc_out,
                'hidden_dim': hidden_dim, # lstm hidden dim
                'linear_hidden_dim': linear_hidden_dim, # lstm decoder最后linear层hidden
                'pred_len': forecast_horizon,
                'fine_tune_layer_dim': fine_tune_layer_dim
            }
            
            model = LSTMEncoderNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_lstm':
            # metaTS^2Net LSTM算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 16 # warm up时候的batchsize
            smoothing_factor = 0.2

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            hidden_dim = 256 # lstm hidden dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # mesta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 6
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'lstm_hidden_dim': hidden_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentLSTM(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_dlinear':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'fine_tune_layer_dim': fine_tune_layer_dim,
                'kernel_size': kernel_size
            }
            
            model = DLinearNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_dlinear':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0.2
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 1
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'z_dim': z_dim,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'latent_dim': latent_dim,
                'device': device,
                'kernel_size': kernel_size,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentDLinear(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,                
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers

                

            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_patch':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2

            # 用于打印
            model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.0,
                'head_dropout': 0.0,
            }
            
            model = PatchtstNaive(**model_config).to(device)
            env_model = OnlineEnvModel
    
        elif algorithm == 'meta_patch':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 32 # 预测模型latent
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            smoothing_factor = 0.5
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 16
            meta_loss_dim = latent_dim * 16
            latent_meta_loss_dim = latent_dim * 16
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 16
            extrapolation_dim = latent_dim * 16
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 8
            sim_loss_use_alpha = True
            sim_loss_score_num = 10

            # 用于打印
            target_model_config = {
                'context_window': look_back_window,
                'd_model': 16,
                'target_window': forecast_horizon,
                'c_in': enc_in,
                'c_out': enc_out,
                'patch_len': 16,
                'stride': 8,
                'n_layers': 3,
                'n_heads': 16,
                'd_ff': 256,
                'revin': 1,
                'affine': 0,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'linear_hidden_dim': linear_hidden_dim,
                'latent_dim': latent_dim,
                'device': device,
                'param_generator_dim': param_generator_dim,
            }
            
            target_model = LatentPatchtst(**target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model
        
        elif algorithm == 'naive_tcn':
            # baseline mlp算法参数, 
            # 没用的参数，用于保持一致
            inner_step = None # meta的参数，没用
            inner_lr = None # meta的参数，没用
            latent_dim = None # meta的参数，没用
            num_epochs_update = None # meta的参数，没用
            z_dim = None # z dim，没用
            num_episode = 7 # LSTM输入过去多少个dataset or epiosde用来预测未来，没用

            #这些是有用的
            num_epochs = 1 # online更新次数
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            linear_hidden_dim = 512 # mlp hidden
            meta_train_epochs = 0  # meta train循环训练多少次
            fine_tune_layer_dim = 128 # 最后一层fine tune layer (output layer) 的dim
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            reg_latent_coef = 0.1
            latent_wd = 0.2
            

            # 用于打印
            model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels
                
            }
            
            model = TCNNaive(**model_config).to(device)
            env_model = OnlineEnvModel
            
        elif algorithm == 'meta_tcn':
            # metaTS^2Net mlp算法参数, 
            inner_step = 50 # adapt 时候epoch数
            inner_lr = 0.001 # adapt时候lr
            num_epochs = 1 # 在meta train阶段更新meta learner的epochs
            num_epochs_update = 1 # 在meta test阶段更新meta learner的epochs
            latent_dim = 128 # 预测模型latent
            num_episode = 5 # LSTM输入过去多少个dataset or epiosde用来预测未来
            warm_up_epochs = 10 # warm up 多少个epoch
            warm_up_batch_size = 32 # warm up时候的batchsize
            smoothing_factor = 0.1

            optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            warm_up_optim = (torch.optim.Adam, {'lr': 1e-3, 'weight_decay': 0})
            batch_size = 32 # 非warm up batchsize
            shuffle = True 
            loss = nn.MSELoss()
            metrics = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}
            z_dim = 128 # z dim
            linear_hidden_dim = 512 # lstm decoder最后linear层hidden
            meta_train_epochs = 1 # meta train循环训练多少次
            kernel_size = 25
            num_channels = [16, 16, 16]
            drop_out = 0.1
            warm_up_mode = None
            mask_percentage = 0.5
            beta_meta_loss = 1
            beta_sim_loss = 1
            beta_latent_meta_loss = 1
            beta_latent_sim_loss = 0
            reg_latent_coef = 0.1
            latent_wd = 0
            sim_loss_dim = latent_dim * 8
            meta_loss_dim = latent_dim * 8
            latent_meta_loss_dim = latent_dim * 6
            latent_meta_loss_layers = 2
            param_generator_dim = latent_dim * 8
            extrapolation_dim = latent_dim * 8
            adapt_latent_lr = 0.001
            adapt_sample_lr = 0.001
            extrapolation_linear_dim = latent_dim * 6
            sim_loss_use_alpha = True
            sim_loss_score_num = 10
            
            target_model_config = {
                'seq_len': look_back_window,
                'pred_len': forecast_horizon,
                'enc_in': enc_in,
                'enc_out': enc_out,
                'linear_hidden_dim': linear_hidden_dim,
                'kernel_size': kernel_size,
                'drop_out': drop_out,
                'num_channels': num_channels,
                'device': device,
                'latent_dim': latent_dim,
                'param_generator_dim': param_generator_dim
            }
            # 定义target model
            target_model = LatentTCN(target_model_config).to(device)
            # 定义metaTS^2Net config
            model_config = {
                'num_episode': num_episode,
                'latent_dim': latent_dim,
                'inner_lr': inner_lr,
                'inner_step': inner_step,
                'memory_size': 7 * 96,
                'target_model_config': target_model_config,
                'device': device,
                'smoothing_factor': smoothing_factor,
                'mask_percentage': mask_percentage,
                'beta_meta_loss': beta_meta_loss,
                'beta_sim_loss': beta_sim_loss,
                'latent_wd': latent_wd,
                'sim_loss_dim': sim_loss_dim,
                'meta_loss_dim': meta_loss_dim,
                'param_generator_dim': param_generator_dim,
                'extrapolation_dim': extrapolation_dim,
                'adapt_latent_lr': adapt_latent_lr,
                'adapt_sample_lr': adapt_sample_lr,
                'extrapolation_linear_dim': extrapolation_linear_dim,
                'sim_loss_use_alpha': sim_loss_use_alpha,
                'sim_loss_score_num': sim_loss_score_num,
                'beta_latent_meta_loss': beta_latent_meta_loss,
                'beta_latent_sim_loss': beta_latent_sim_loss,
                'latent_meta_loss_dim': latent_meta_loss_dim,
                'latent_meta_loss_layers': latent_meta_loss_layers


            }
            
            model = MetaTS2Net(target_model, model_config).to(device)
            env_model = OnlineMeta2Model

            
        # training env setting,也用于保存所以包含所有参数
        configs = {
            'model': model,
            'num_episode': num_episode,
            'latent_dim': latent_dim,
            'inner_lr': inner_lr,
            'inner_step': inner_step,
            'optim': optim,
            'warm_up_optim': warm_up_optim,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'loss': loss,
            'z_dim': z_dim,
            'warm_up_batch_size': warm_up_batch_size,
            'warm_up_epochs': warm_up_epochs,
            'metrics': metrics,
            'num_epochs': num_epochs,
            'num_epochs_update': num_epochs_update,
            'meta_train_epochs': meta_train_epochs,
            'meta_train_length': meta_train_length,
            'look_back_window': look_back_window, # input_len
            'forecast_horizon': forecast_horizon, # ouptut_len
            'total_time_interval': total_time_interval, #总共时间长度, online_env parameters
            'warm_up_interval': warm_up_interval, # warm up 范围
            'online_step_size': online_step_size, # online过程中每个dataset的长度
            'split_times': split_times,
            'model_config': model_config,
            'warm_up_mode': warm_up_mode,
            'latent_wd': latent_wd,
            'reg_latent_coef': reg_latent_coef
        }
        
        env = OnlineEnv(split_times=split_times, metrics=metrics, datasets=(train, test))
        train_cls = OnlineEnvTrain(env, env_model, deepcopy(configs))

    return train_cls, configs

def plot_and_save(train_cls_dict, path, fh_dict):
    # 画图+保存
    for name, train_cls_lst in train_cls_dict.items():
        fh = fh_dict[name]
        for j, train_cls in enumerate(train_cls_lst):
            true_df = train_cls.true_df
            pred_df = train_cls.pred_df
            out_true_lst = []
            out_pred_lst = []
            for i in range(0, len(true_df), fh):
                true = true_df.iloc[i].tolist()
                pred = pred_df.iloc[i].tolist()
                out_true_lst.extend(true)
                out_pred_lst.extend(pred)

            df = pd.DataFrame(np.array([out_true_lst, out_pred_lst]).T, columns=['true', 'pred'])
            plot_obj = df.plot(figsize=(20, 10)).get_figure()
            plot_obj.savefig(f'{path}/{name}_{j}.png')

            plot_obj_s = df[-14*96:].plot(figsize=(20, 10)).get_figure()
            plot_obj_s.savefig(f'{path}/{name}_{j}_s.png')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['bus_1']) # etth2, etthm1, ecl, bus_1, bus_2, bus_3
    parser.add_argument('--algorithms', type=str, nargs='+', default=['naive_lstm']) # naive_lstm, meta_lstm, naive_linear, meta_linear
    parser.add_argument('--mode', type=str, default='fine_tune') # fine_tune, naive, meta_finetune
    parser.add_argument('--rep', type=int, default=1) # 每个跑几次？
    parser.add_argument('--plot', type=bool, default=True) # 是不是要画预测图？
    parser.add_argument('--data_dir', type=str, default='./datasets') # 数据放的位置
    parser.add_argument('--save_dir_name', type=str, default=None) # 结果存放的文件名字如“sample_adap_false_epoch_3”
    parser.add_argument('--seed', type=int, default=16)
    args = parser.parse_args()
    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    torch.autograd.set_detect_anomaly(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    if args.save_dir_name:
        save_name = args.save_dir_name
    else:
        save_name = start_time

    if not os.path.exists(f'./results/{save_name}'):
        os.mkdir(f'./results/{save_name}')
    
    # config logger
    _ = init_logger(f'./results/{save_name}/log', 'main.py')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fh_dict = {}
    out_mean_acc_dict = {}
    for dataset in args.datasets:
        for algorithm in args.algorithms:
            if args.mode == 'meta_finetune':
                train_cls, configs = fetch_train_cls(dataset=dataset, algorithm=algorithm, mode='fine_tune', device=device, data_dir=args.data_dir)
            else:
                train_cls, configs = fetch_train_cls(dataset=dataset, algorithm=algorithm, mode=args.mode, device=device, data_dir=args.data_dir)
            
            n = f'{dataset}_{algorithm}'
            if n not in fh_dict:
                fh_dict[n] = configs['forecast_horizon']
            
            logging.info(configs)
            out_train_cls, mean_acc_dict = score_all([(train_cls, n)], args.mode, args.rep, seed=args.seed)
            out_mean_acc_dict.update(mean_acc_dict)

            if args.plot:
                plot_and_save(out_train_cls, f'./results/{save_name}', fh_dict)
    
    with open(f'./results/{save_name}/acc.pkl', 'w') as f:
        json.dump(out_mean_acc_dict, f)
    
    

if __name__ == '__main__':
    main()