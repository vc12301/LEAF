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


# TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNNaive(nn.Module):
    def __init__(self, 
                 enc_in,
                 enc_out,
                 linear_hidden_dim,
                 pred_len,
                 seq_len,
                 kernel_size,
                 drop_out,
                 num_channels):
        super().__init__()
        self.enc_in = enc_in
        self.linear_hidden_dim = linear_hidden_dim
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.drop_out = drop_out
        self.num_channels = num_channels
        self.enc_out = enc_out
        self.num_channels = self.num_channels + [self.enc_out]
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)
        
        self.tcn = TemporalConvNet(enc_in, num_channels, self.kernel_size, self.drop_out)

        self.fine_tune_layer = nn.ModuleList()
        for i in range(self.enc_in):
            self.fine_tune_layer.append(nn.Linear(seq_len, pred_len))
    
    def forward(self, X):
        N, L, C = X.shape
        X = self.revin_layer(X, 'norm')
        X = X.permute(0, 2, 1)
        h = self.tcn(X)
        # h.shape is (N, num_channel[-1], seq_len)
        output = torch.zeros(N, C, self.pred_len).to(X.device)
        for i in range(self.enc_in):
            output[:, i,:] = self.fine_tune_layer[i](h[:, i, :])

        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        return output
    
    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions
    

class TCNEncoder(nn.Module):
    def __init__(self, 
                 enc_in,
                 enc_out,
                 linear_hidden_dim,
                 pred_len,
                 seq_len,
                 kernel_size,
                 drop_out,
                 num_channels):
        super().__init__()
        self.enc_in = enc_in
        self.linear_hidden_dim = linear_hidden_dim
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.drop_out = drop_out
        self.num_channels = num_channels
        self.enc_out = enc_out
        self.num_channels = self.num_channels + [self.enc_out]
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)
        
        self.tcn = TemporalConvNet(enc_in, num_channels, self.kernel_size, self.drop_out)
    
    def forward(self, X):
        N, L, C = X.shape
       #  X = self.revin_layer(X, 'norm')
        X = X.permute(0, 2, 1)
        h = self.tcn(X)


        # output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        return h
    
    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions

class LatentTCN(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super().__init__()
        configs = SimpleNamespace(**configs)
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # num of channels
        self.enc_out = configs.enc_out
        self.linear_hidden_dim = configs.linear_hidden_dim
        self.kernel_size = configs.kernel_size
        self.drop_out = configs.drop_out
        self.num_channels = configs.num_channels
        self.latent_dim = configs.latent_dim
        self.device = configs.device
        self.param_generator_dim = configs.param_generator_dim
        
        self.g_tcn = TCNEncoder(enc_in=self.enc_in, 
                                linear_hidden_dim=self.linear_hidden_dim,
                                seq_len=self.seq_len,
                                pred_len=self.pred_len,
                                drop_out=self.drop_out,
                                num_channels=self.num_channels,
                                kernel_size=self.kernel_size,
                                enc_out=self.enc_out)
        
        self.latent = self._init_tensor(torch.zeros(self.enc_out, self.latent_dim)).to(self.device)
        self.latent.requires_grad_()
        
        self.param_generator = nn.ModuleList()
        # # self.g_mlp = nn.ModuleList()
        for i in range(self.enc_in):
            self.param_generator.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.param_generator_dim, bias=True),
                    nn.ReLU(),
                    # nn.Linear(self.latent_dim * 8, self.latent_dim * 8, bias=True),
                    # nn.ReLU(),
                    # nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
                    # nn.ReLU(),
                    nn.Linear(self.param_generator_dim, self.seq_len * self.pred_len + self.pred_len, bias=True)))

    def forward(self, X, sample_latents=None):
        # x: [Batch, Input length, Channel]
        N, L, C = X.shape
        output = torch.zeros(N, C, self.pred_len).to(self.device)
        # X = self.revin_layer(X, 'norm')
        z = self.g_tcn(X)

        if sample_latents is not None:
            for i in range(C):
                self.params = self.param_generator[i](sample_latents[:, i, :]) # N * self.seq_len
                weight = self.params[:, :self.seq_len * self.pred_len].reshape(N, self.seq_len, self.pred_len)
                bias = self.params[:, self.seq_len * self.pred_len:].reshape(N, self.pred_len)
                output[:, i, :] = torch.einsum('nz, nzp -> np', [z[:, i, :], weight]) + bias
        else:
            for i in range(C):
                self.params = self.param_generator[i](self.latent[i, :])
                weight = self.params[:self.seq_len * self.pred_len].reshape(self.seq_len, self.pred_len)
                bias = self.params[self.seq_len * self.pred_len:]
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

    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions