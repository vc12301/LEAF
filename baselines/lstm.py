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


class LSTMEncoder(nn.Module):
    def __init__(self,
                 enc_in,
                 hidden_dim,
                 linear_hidden_dim,
                 n_layers=1,
                 batch_first=True,
                 fine_tune_layer_dim=128):
        super().__init__()

        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        self.linear_hidden_dim = linear_hidden_dim
        self.n_layers = n_layers
        self.mlp_decoder = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, fine_tune_layer_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=hidden_dim, num_layers=n_layers, batch_first=batch_first)

    def forward(self, X):
        N, L, C = X.shape
        p, (h, c) = self.lstm(X)
        # h.shape is (N, self.n_layers, self.hidden_dim)
        # flatten to (N, self.n_layers * self.hidden_dim)
        h = h.reshape(N, -1)
        z = self.mlp_decoder(h)
    
        return z

    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions

    
class LSTMEncoderNaive(nn.Module):
    def __init__(self,
                 enc_in,
                 hidden_dim,
                 linear_hidden_dim,
                 pred_len,
                 n_layers=1,
                 enc_out=1,
                 batch_first=True, 
                 fine_tune_layer_dim=128):
        super().__init__()

        self.enc_in = enc_in
        self.enc_out = enc_out
        self.hidden_dim = hidden_dim
        self.linear_hidden_dim = linear_hidden_dim
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.mlp_decoder = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, fine_tune_layer_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=hidden_dim, num_layers=n_layers, batch_first=batch_first)
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)

        self.fine_tune_layer = nn.ModuleList()
        for i in range(self.enc_in):
            self.fine_tune_layer.append(nn.Linear(fine_tune_layer_dim, pred_len))

    def forward(self, X):
        N, L, C = X.shape
        X = self.revin_layer(X, 'norm')
        p, (h, c) = self.lstm(X)
        # h.shape is (N, self.n_layers, self.hidden_dim)
        # flatten to (N, self.n_layers * self.hidden_dim)
        h = h.reshape(N, -1)
        z = self.mlp_decoder(h)

        output = torch.zeros(N, C, self.pred_len).to(z.device)
        for i in range(self.enc_in):
            output[:, i,:] = self.fine_tune_layer[i](z)

        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        # z.shape is (N, pred_len)
        # output = output.permute(0, 2, 1)
        
        return output

    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions
    
    
class LatentLSTM(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super().__init__()
        configs = SimpleNamespace(**configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.z_dim = configs.z_dim # z dim
        self.lstm_hidden_dim = configs.lstm_hidden_dim
        self.enc_in = configs.enc_in # num of channels
        self.enc_out = configs.enc_out # num of out channels
        self.linear_hidden_dim = configs.linear_hidden_dim
        self.latent_dim = configs.latent_dim # H dim
        self.device = configs.device
        self.param_generator_dim = configs.param_generator_dim
        
        self.g_lstm = LSTMEncoder(enc_in=self.enc_in,
                                  hidden_dim=self.lstm_hidden_dim,
                                  linear_hidden_dim=self.linear_hidden_dim,
                                  n_layers=1,
                                  batch_first=True,
                                  fine_tune_layer_dim=self.z_dim)
        
        self.latent = self._init_tensor(torch.zeros(self.enc_out, self.latent_dim)).to(self.device)
        self.latent.requires_grad_()
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)

        self.param_generator = nn.ModuleList()
        for i in range(self.enc_in):
            self.param_generator.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.param_generator_dim, bias=True),
                    nn.ReLU(),
                    # nn.Linear(self.latent_dim * 8, self.latent_dim * 8, bias=True),
                    # nn.ReLU(),
                    # nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
                    # nn.ReLU(),
                    nn.Linear(self.param_generator_dim, self.z_dim * self.pred_len + self.pred_len, bias=True)))
        
    def forward(self, X, sample_latents=None):
        # x: [Batch, Input length, Channel]
        N, L, C = X.shape
        output = torch.zeros(N, C, self.pred_len).to(self.device)
        output_1 = torch.zeros(N, C, self.pred_len).to(self.device)
        # X = self.revin_layer(X, 'norm')
        z = self.g_lstm(X)

        if sample_latents is not None:
            for i in range(C):
                self.params = self.param_generator[i](sample_latents[:, i, :]) # N * self.latent_dim
                weight = self.params[:, :self.z_dim * self.pred_len].reshape(N, self.z_dim, self.pred_len)
                bias = self.params[:, self.z_dim * self.pred_len:].reshape(N, self.pred_len)
                output[:, i, :] = torch.einsum('nz, nzp -> np', [z, weight]) + bias

        else:
            for i in range(C):
                self.params = self.param_generator[i](self.latent[i, :])
                weight = self.params[:self.z_dim * self.pred_len].reshape(self.z_dim, self.pred_len)
                bias = self.params[self.z_dim * self.pred_len:]
                output[:, i, :] = self.Linear(z, weight, bias)


        # output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        output = output.permute(0, 2, 1)
        return output, z


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
        

    def predict(self, X):
        return self.forward(X)
