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


# class MLPEncoderNaive(nn.Module):
#     def __init__(self,
#                  seq_len,
#                  enc_in,
#                  linear_hidden_dim,
#                  pred_len,
#                  enc_out, 
#                  fine_tune_layer_dim=128):
#         super().__init__()

#         self.seq_len = seq_len
#         self.enc_in = enc_in
#         self.linear_hidden_dim = linear_hidden_dim
#         self.pred_len = pred_len
#         self.enc_out = enc_out
#         self.mlp_decoder = nn.Sequential(
#             nn.Linear(seq_len * enc_in, linear_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(linear_hidden_dim, fine_tune_layer_dim),
#             nn.ReLU(),
#         )
#         self.fine_tune_layer = nn.Linear(fine_tune_layer_dim, pred_len * enc_out)

#     def forward(self, X):
#         N, L, C = X.shape
#         # flatten to (N, L * C)
#         z = self.mlp_decoder(X.reshape(N, L * C))
#         # z.shape is (N, pred_len)
#         # z = z.reshape(N, self.pred_len)
#         z = self.fine_tune_layer(z)
#         z = z.reshape(N, self.pred_len, self.enc_out)
        
#         return z

#     def predict(self, X):
#         with torch.no_grad():
#             predictions = self.forward(X)

#         return predictions


# class LatentMLP(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         configs = SimpleNamespace(**configs)
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.z_dim = configs.z_dim # z dim
#         self.enc_in = configs.enc_in # num of channels
#         self.enc_out = configs.enc_out
#         self.linear_hidden_dim = configs.linear_hidden_dim
#         self.latent_dim = configs.latent_dim
#         self.device = configs.device
        
#         self.g_mlp = nn.Sequential(
#             nn.Linear(self.seq_len * self.enc_in, self.linear_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.linear_hidden_dim, self.z_dim),
# #            nn.ReLU(),
#         )
        
#         self.latent_dim = configs.latent_dim # H dim
#         self.latent = self._init_tensor(torch.zeros(self.latent_dim,)).to(self.device)
#         self.latent.requires_grad_()
        
#         # \omega
#         self.param_generator = nn.Sequential(
#             nn.Linear(self.latent_dim, self.latent_dim * 4, bias=True),
#             nn.ReLU(),
#             nn.Linear(self.latent_dim * 4, self.latent_dim * 8, bias=True),
#             nn.ReLU(),
#             nn.Linear(self.latent_dim * 8, self.latent_dim * 8, bias=True),
#             nn.ReLU(),
#             nn.Linear(self.latent_dim * 8, (self.z_dim * self.pred_len + self.pred_len) * self.enc_out, bias=True)
#         )

#     def forward(self, X):
#         # x: [Batch, Input length, Channel]
#         N, L, C = X.shape
#         z = self.g_mlp(X.reshape(N, L * C)).reshape(N, -1)
#         # z: [N, self.z_dim]

#         out = self.forward_z(z)
        
#         return out, z

#     def forward_z(self, z):
#         # z: [N, ]
#         # decode parameter using omega(H) -> weight, bias
#         N, _ = z.shape
#         self.params = self.param_generator(self.latent)
#         # f
#         weight = self.params[:self.z_dim * self.pred_len * self.enc_out].reshape(self.z_dim, self.pred_len * self.enc_out)
#         bias = self.params[self.z_dim * self.pred_len * self.enc_out:]
#         out = self.Linear(z, weight, bias).reshape(N, self.pred_len, self.enc_out)
        
#         return out # [Batch, Output length, Channel]

#     def _init_tensor(self, tensor):
#         dim = tensor.shape[-1]
#         std = 1 / math.sqrt(dim)
#         tensor.uniform_(-std, std)
#         return tensor

#     def Linear(self, x, weight, bias):
#         if len(bias.shape) == 1:
#             return torch.matmul(x, weight) + bias
#         else:
#             return torch.matmul(x, weight) + bias.unsqueeze(1)
        
#     def set_mode(self, mode):
#         assert mode in ['warmup', 'finetune']
#         if mode == 'warmup':
#             for name, param in self.named_parameters():
#                 if name == 'latent':
#                     param.requires_grad = False
#                 else:
#                     param.requires_grad = True
#         elif mode == 'finetune':
#             for name, param in self.named_parameters():
#                 if name == 'latent':
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False


class MLPEncoderNaive(nn.Module):
    def __init__(self,
                 seq_len,
                 enc_in,
                 linear_hidden_dim,
                 pred_len,
                 enc_out, 
                 fine_tune_layer_dim=128):
        super().__init__()
        self.revin_layer = RevIN(enc_in, affine=False, subtract_last=False)
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.linear_hidden_dim = linear_hidden_dim
        self.pred_len = pred_len
        self.enc_out = enc_out
        self.mlp_decoder = nn.Sequential(
            nn.Linear(seq_len, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, fine_tune_layer_dim),
            nn.ReLU(),
        )
        self.fine_tune_layer_dim = fine_tune_layer_dim
        # self.fine_tune_layer = nn.Linear(fine_tune_layer_dim, pred_len)

        # self.mlp_decoder = nn.ModuleList()
        self.fine_tune_layer = nn.ModuleList()
        for i in range(self.enc_in):
            self.fine_tune_layer.append(nn.Linear(fine_tune_layer_dim, pred_len))
        

    def forward(self, X):
        N, L, C = X.shape
        # if self.if_fine_tune_layer:
        #     out = torch.zeros(N, C, self.pred_len).to(X.device)
        # else:
        #     out = torch.zeros(N, C, self.fine_tune_layer_dim).to(X.device)

        # X = X.permute(0, 2, 1)
        # for i in range(self.enc_in):
        #     z = self.mlp_decoder[i](X[:, i, :])
        #     if self.if_fine_tune_layer:
        #         z = self.fine_tune_layer[i](z)

        #     out[:, i, :] = z

        X = self.revin_layer(X, 'norm')
        z = self.mlp_decoder(X.permute(0, 2, 1))
        output = torch.zeros(N, C, self.pred_len).to(z.device)
        for i in range(self.enc_in):
            output[:, i,:] = self.fine_tune_layer[i](z[:, i, :])
        # z.shape is (N, enc_out, fine_tune_layer_dim)

        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')
        
        # out = out.permute(0, 2, 1)
        return output

    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions


class LatentMLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs = SimpleNamespace(**configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in # num of channels
        self.enc_out = configs.enc_out
        self.linear_hidden_dim = configs.linear_hidden_dim
        self.latent_dim = configs.latent_dim
        self.device = configs.device
        self.z_dim = configs.z_dim
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)
        
        self.g_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_dim, self.z_dim),
            nn.ReLU(),
        )
        
        self.latent_dim = configs.latent_dim # H dim
        self.latent = self._init_tensor(torch.zeros(self.enc_out, self.latent_dim)).to(self.device)
        self.latent.requires_grad_()
        
        self.param_generator = nn.ModuleList()
        # # self.g_mlp = nn.ModuleList()
        for i in range(self.enc_in):
            self.param_generator.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim * 8, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim * 8, self.latent_dim * 8, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim * 8, self.latent_dim * 16, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim * 16, self.z_dim * self.pred_len + self.pred_len, bias=True)))
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
        
    def forward(self, X):
        # x: [Batch, Input length, Channel]
        N, L, C = X.shape
        output = torch.zeros(N, C, self.pred_len).to(self.device)
        X = self.revin_layer(X, 'norm')
        z = self.g_mlp(X.permute(0, 2, 1))
        for i in range(C):
            self.params = self.param_generator[i](self.latent[i, :])
            weight = self.params[:self.z_dim * self.pred_len].reshape(self.z_dim, self.pred_len)
            bias = self.params[self.z_dim * self.pred_len:]
            output[:, i, :] = self.Linear(z[:, i, :], weight, bias)

        output = self.revin_layer(output.permute(0, 2, 1), 'denorm')

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