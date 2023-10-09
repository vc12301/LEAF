import argparse
import os
import pandas as pd
import torch.nn as nn
from baselines.layers import RevIN, TSTiEncoder
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
from typing import Callable, Optional
from torch import Tensor


class PatchtstNaive(nn.Module):
    def __init__(self, context_window:int, c_in:int, d_model:int, target_window:int, c_out, patch_len:int=16, stride:int=8, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, channel_cross=False,
                 verbose:bool=False, mix_tcn=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, mix_tcn=mix_tcn, **kwargs)

        # Head
        self.channel_cross = channel_cross
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.pred_len = target_window

        self.flatten = nn.Flatten(start_dim=-2)

        self.fine_tune_layer = nn.ModuleList()
        for i in range(self.n_vars):
            self.fine_tune_layer.append(nn.Linear(self.head_nf, target_window))

        # self.fine_tune_layer = nn.Linear(self.head_nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        z = z.permute(0,2,1)

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]

        z = self.flatten(z)
        N, C, _ = z.shape
        output = torch.zeros(N, C, self.pred_len).to(z.device)
        for i in range(self.n_vars):
            output[:, i,:] = self.fine_tune_layer[i](z[:, i, :])
        
        output = self.dropout(output)
        
        # denorm
        if self.revin: 
            output = output.permute(0,2,1)
            output = self.revin_layer(output, 'denorm')
            output = output.permute(0,2,1)

        return output.permute(0,2,1)


    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions



class LatentPatchtst(nn.Module):
    def __init__(self, context_window:int, c_in:int, d_model:int, target_window:int, c_out, linear_hidden_dim, latent_dim, device, patch_len:int=16, stride:int=8, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = False, affine = True, subtract_last = False, channel_cross=False,
                 verbose:bool=False, param_generator_dim=4096, **kwargs):
        
        super().__init__()
        self.seq_len = context_window
        self.pred_len = target_window
        self.enc_in = c_in
        self.enc_out = c_in
        self.linear_hidden_dim = linear_hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.param_generator_dim = param_generator_dim
        
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        self.z_dim = d_model * patch_num
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.channel_cross = channel_cross
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.head_nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        
        self.latent = self._init_tensor(torch.zeros(self.enc_in, self.latent_dim)).to(self.device)
        self.latent.requires_grad_()
        
        # \omega
        self.param_generator = nn.ModuleList()
        for i in range(self.enc_in):
            self.param_generator.append(nn.Sequential(
                nn.Linear(self.latent_dim, self.param_generator_dim),
                nn.ReLU(),
                nn.Linear(self.param_generator_dim, (self.head_nf + 1) * target_window)
            ))
    
    # def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
    #     # norm
    #     z = z.permute(0,2,1)
    #     # if self.revin: 
    #     #     z = z.permute(0,2,1)
    #     #     z = self.revin_layer(z, 'norm')
    #     #     z = z.permute(0,2,1)
    #     # do patching
    #     if self.padding_patch == 'end':
    #         z = self.padding_patch_layer(z)
    #     z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
    #     z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
    #     # model
    #     z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
    #     z = self.flatten(z)
    #     N, C, _ = z.shape
    #     output = torch.zeros(N, C, self.pred_len).to(z.device)
    #     for i in range(C):
    #         self.params = self.param_generator[i](self.latent[i, :])
    #         weight = self.params[:self.head_nf * self.pred_len].reshape(self.head_nf, self.pred_len)
    #         bias = self.params[self.head_nf * self.pred_len:]
    #         output[:, i, :] = self.Linear(z[:, i, :], weight, bias)
        
    #     output = self.dropout(output)

    #     # if self.revin: 
    #     #     output = output.permute(0, 2, 1)
    #     #     output = self.revin_layer(output, 'denorm')
    #     # else:
    #     output = output.permute(0, 2, 1)
     
    #     return output, z

    
    def forward(self, z, sample_latents=None):                                                                   # z: [bs x nvars x seq_len]
        # norm
        z = z.permute(0,2,1)
        # if self.revin: 
        #     z = z.permute(0,2,1)
        #     z = self.revin_layer(z, 'norm')
        #     z = z.permute(0,2,1)
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.flatten(z)
        N, C, _ = z.shape
        output = torch.zeros(N, C, self.pred_len).to(z.device)

        if sample_latents is not None:
            for i in range(C):
                self.params = self.param_generator[i](sample_latents[:, i, :]) # N * self.latent_dim
                weight = self.params[:, :self.head_nf * self.pred_len].reshape(N, self.head_nf, self.pred_len)
                bias = self.params[:, self.head_nf * self.pred_len:].reshape(N, self.pred_len)
                output[:, i, :] = torch.einsum('nz, nzp -> np', [z[:, i, :], weight]) + bias
        else:
            for i in range(C):
                self.params = self.param_generator[i](self.latent[i, :])
                weight = self.params[:self.head_nf * self.pred_len].reshape(self.head_nf, self.pred_len)
                bias = self.params[self.head_nf * self.pred_len:]
                output[:, i, :] = self.Linear(z[:, i, :], weight, bias)

        
        output = self.dropout(output)

        # if self.revin: 
        #     output = output.permute(0, 2, 1)
        #     output = self.revin_layer(output, 'denorm')
        # else:
        output = output.permute(0, 2, 1)
     
        return output, z


    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions
    
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