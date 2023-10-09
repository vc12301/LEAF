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
from pprint import pprint
from baselines.layers import RevIN


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearNaive(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, fine_tune_layer_dim=128, kernel_size=25, enc_out=7):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_out = enc_out
        self.fine_tune_layer_dim = fine_tune_layer_dim
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.fine_tune_layer_dim)
        self.Linear_Trend = nn.Linear(self.seq_len, self.fine_tune_layer_dim)
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.enc_in = enc_in
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)

        self.fine_tune_layer = nn.ModuleList()
        for i in range(self.enc_in):
            self.fine_tune_layer.append(nn.Linear(self.fine_tune_layer_dim, pred_len))
        
    def forward(self, x):
        N, L, C = x.shape

        x = self.revin_layer(x, 'norm')
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output # N, C, L

        output = torch.zeros(N, C, self.pred_len).to(x.device)
        for i in range(self.enc_in):
            output[:, i,:] = self.fine_tune_layer[i](x[:, i, :])

        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')

        return output # to [Batch, Output length, Channel]
    
    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions


class DLinearEncoder(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.enc_in = enc_in
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.decompsition = series_decomp(kernel_size)
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)
        
    def forward(self, x):
        N, L, C = x.shape

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output # N, C, L

        return x 
    
    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions


class LatentDLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs = SimpleNamespace(**configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # num of channels
        self.enc_out = configs.enc_out
        # self.linear_hidden_dim = configs.linear_hidden_dim
        self.latent_dim = configs.latent_dim
        self.device = configs.device
        self.z_dim = configs.z_dim
        self.kernel_size = configs.kernel_size
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)
        self.param_generator_dim = configs.param_generator_dim
        
        self.g_mlp = DLinearEncoder(self.seq_len, self.z_dim, self.enc_in, self.kernel_size)
        
        self.latent_dim = configs.latent_dim # H dim
        self.latent = self._init_tensor(torch.zeros(self.enc_out, self.latent_dim)).to(self.device)
        self.latent.requires_grad_()
        
        self.param_generator = nn.ModuleList()
        # # self.g_mlp = nn.ModuleList()

        # This is the setting for ninjing_2
        for i in range(self.enc_in):
            self.param_generator.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.param_generator_dim, bias=True),
                    nn.ReLU(),
                    # nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
                    # nn.ReLU(),
                    nn.Linear(self.param_generator_dim, self.z_dim * self.pred_len + self.pred_len, bias=True)))
            # self.g_mlp.append(
            #     MLPEncoderNaive(self.seq_len, self.enc_in, self.linear_hidden_dim, self.pred_len, self.enc_out, if_fine_tune_layer=False, fine_tune_layer_dim=self.z_dim)
            # )

        # for i in range(self.enc_in):
        #     self.param_generator.append(
        #         nn.Sequential(
        #             nn.Linear(self.latent_dim, self.latent_dim * 8, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 16, self.latent_dim * 32, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 32, self.z_dim * self.pred_len + self.pred_len, bias=True)))
            # self.g_mlp.append(
            #     MLPEncoderNaive(self.seq_len, self.enc_in, self.linear_hidden_dim, self.pred_len, self.enc_out, if_fine_tune_layer=False, fine_tune_layer_dim=self.z_dim)
            # )


        # \omega
        # self.param_generator = nn.Sequential(
        #             nn.Linear(self.latent_dim, self.latent_dim * 8, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 8, self.latent_dim * 8, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
        #             nn.ReLU(),
        #             nn.Linear(self.latent_dim * 16, self.z_dim * self.pred_len + self.pred_len, bias=True)
        # )
        
    def forward(self, X, sample_latents=None):
        # x: [Batch, Input length, Channel]
        N, L, C = X.shape
        output = torch.zeros(N, C, self.pred_len).to(self.device)
        # X = self.revin_layer(X, 'norm')
        z = self.g_mlp(X)

        if sample_latents is not None:
            for i in range(C):
                self.params = self.param_generator[i](sample_latents[:, i, :]) # N * self.latent_dim
                weight = self.params[:, :self.z_dim * self.pred_len].reshape(N, self.z_dim, self.pred_len)
                bias = self.params[:, self.z_dim * self.pred_len:].reshape(N, self.pred_len)
                output[:, i, :] = torch.einsum('nz, nzp -> np', [z[:, i, :], weight]) + bias
        else:
            for i in range(C):
                self.params = self.param_generator[i](self.latent[i, :])
                weight = self.params[:self.z_dim * self.pred_len].reshape(self.z_dim, self.pred_len)
                bias = self.params[self.z_dim * self.pred_len:]
                output[:, i, :] = self.Linear(z[:, i, :], weight, bias)

        # output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        output = output.permute(0, 2, 1)
        return output, z
    # def forward(self, X):
    #     # x: [Batch, Input length, Channel]
    #     N, L, C = X.shape
    #     output = torch.zeros(N, C, self.pred_len).to(self.device)
    #     z = self.g_mlp(X).permute(0, 2, 1) # N, z_dim, C
    #     self.params = self.param_generator(self.latent)
    #     for i in range(C):
    #         weight = self.params[i, :self.z_dim * self.pred_len].reshape(self.z_dim, self.pred_len)
    #         bias = self.params[i, self.z_dim * self.pred_len:]
    #         output[:, i, :] = self.Linear(z[:, i, :], weight, bias)

    #     # out = self.forward_z(z.permute(0, 2, 1))
    #     # out = out.permute(0, 2, 1) # N, L, C
    #     output = output.permute(0, 2, 1)

    #     return output, z
    # def forward_z(self, z):
    #     # z: [N, C, L]
    #     # decode parameter using omega(H) -> weight, bias
    #     N, C, z_dim = z.shape
    #     # C * latent_dim 
    #     output = torch.zeros(N, C, self.pred_len).to(self.device)
    #     for i in range(C):
    #         self.params = self.param_generator[i](self.latent[i, :])
    #         weight = self.params[:self.z_dim * self.pred_len].reshape(self.z_dim, self.pred_len)
    #         bias = self.params[self.z_dim * self.pred_len:]
    #         output[:, i, :] = self.Linear(z[:, i, :], weight, bias)

    #     return output

    def _init_tensor(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor

    def Linear(self, x, weight, bias):
        if len(bias.shape) == 1:
            return torch.matmul(x, weight) + bias
        else:
            return torch.matmul(x, weight) + bias.unsqueeze(1)
        
    def set_mode(self, mode):
        assert mode in ['warmup', 'finetune']
        if mode == 'warmup':
            for name, param in self.named_parameters():
                if name == 'latent':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif mode == 'finetune':
            for name, param in self.named_parameters():
                if name == 'latent':
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for p in self.param_generator.parameters():
                p.requires_grad = True




        