import pandas as pd
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from types import SimpleNamespace
import math
from baselines.layers import RevIN

class mod_relu(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, z):
        return self.relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z)) 


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


class ExtrapolationNetwork(nn.Module):
    def __init__(self,
                 enc_in,
                 hidden_dim,
                 linear_hidden_dim,
                 n_layers=1,
                 batch_first=True):
        super().__init__()

        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        self.linear_hidden_dim = linear_hidden_dim
        self.n_layers = n_layers
        self.mlp_decoder = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(linear_hidden_dim, enc_in),
        )

        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=hidden_dim, num_layers=n_layers, batch_first=batch_first)

    def forward(self, H):
        # H.shape = (C, num_episodes, latent_dim)
        p, (h, c) = self.lstm(H)
        # h.shape is (N, self.n_layers, self.hidden_dim)
        out = self.mlp_decoder(h.reshape(H.shape[0], -1))

        return out


    def predict(self, X):
        with torch.no_grad():
            predictions = self.forward(X)

        return predictions

    
# class SimLoss(nn.Module):
#     def __init__(self, input_len, hidden_dim, latent_dim, device, num_sim=10):
#         # meta loss和extrapolation_network一样，都是通过一个LSTM生成一个latent，然后解码这个latent变成Metaloss网络参数，再通过metaloss网络更新z -> z^prime
#         super().__init__()
# #         self.phi = nn.Sequential(
# #             nn.Linear(input_len, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 1)
# #         )
#         self.input_len = input_len # z的长度
#         self.hidden_dim = hidden_dim # meta loss latent长度
#         self.num_sim = num_sim
#         # self.relu = mod_relu(self.input_len)
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim)
#          )
        
#         self.phi_sim = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_sim)
#         )
 
#         self.alphas = nn.Parameter(torch.ones(self.num_sim))
#         self.cos_sim = nn.CosineSimilarity(dim=-1)

#         # self.latent = None
#         # self.latent.requires_grad_()
        
#         # 生成Metaloss网络最后一层参数
#         # self.param_generator = nn.Sequential(
#         #     nn.Linear(self.hidden_dim, self.hidden_dim * 32, bias=True),
#         #     nn.ReLU(),
#         #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4, bias=True),
#         #     # nn.ReLU(),
#         #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8, bias=True),
#         #     # nn.ReLU(),
#         #     nn.Linear(self.hidden_dim * 32, (self.input_len + 1) * self.hidden_dim, bias=True)
#         # )
        
#     def forward(self, X, latent):
#         X_latent = self.phi(X)

#         X_sim = self.phi_sim(X_latent)
#         latent_sim = self.phi_sim(latent)
#         dis = (X_sim - latent_sim) * self.alphas
#         # dis = self.cos_sim(X_sim, latent_sim)
#         # loss = 1 - dis


#         loss = torch.square(torch.norm(dis, p=2, dim=-1, keepdim=True))

        

#         # loss = torch.square(torch.norm(X - latent_adapted, p=2, dim=-1, keepdim=True))
#         # loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
#         # loss = torch.square(self.phi(X))
#         return loss


class SimLoss(nn.Module):
    def __init__(self, input_len, hidden_dim, latent_dim, device, num_sim=10, use_alpha=True):
        # meta loss和extrapolation_network一样，都是通过一个LSTM生成一个latent，然后解码这个latent变成Metaloss网络参数，再通过metaloss网络更新z -> z^prime
        super().__init__()
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        self.input_len = input_len # z的长度
        self.hidden_dim = hidden_dim # meta loss latent长度
        self.num_sim = num_sim
        # self.relu = mod_relu(self.input_len)
        self.phi = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
         )
        

        self.phi_sim = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sim)
        )
        # self.phi_sim = nn.Identity()
        
        self.use_alpha = use_alpha
        self.alphas = nn.Parameter(torch.ones(num_sim))
        # self.cos_sim = nn.CosineSimilarity(dim=-1)

        # self.latent = None
        # self.latent.requires_grad_()
        
        # 生成Metaloss网络最后一层参数
        # self.param_generator = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim * 32, bias=True),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4, bias=True),
        #     # nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8, bias=True),
        #     # nn.ReLU(),
        #     nn.Linear(self.hidden_dim * 32, (self.input_len + 1) * self.hidden_dim, bias=True)
        # )
        
    def forward(self, X, latent):
        X_latent = self.phi(X)

        X_sim = self.phi_sim(X_latent)
        latent_sim = self.phi_sim(latent)
        if self.use_alpha:
            dis = (X_sim - latent_sim) * self.alphas
        else:
            dis = X_sim - latent_sim
        # dis = (X_latent - latent)
        
        # dis = self.cos_sim(X_sim, latent_sim)
        # loss = 1 - dis
        loss = torch.square(torch.norm(dis, p=2, dim=-1, keepdim=True))


        # loss = torch.square(torch.norm(X - latent_adapted, p=2, dim=-1, keepdim=True))
        # loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
        # loss = torch.square(self.phi(X))
        return loss

# class MetaLoss(nn.Module):
#     def __init__(self, input_len, hidden_dim, device):
#         # meta loss和extrapolation_network一样，都是通过一个LSTM生成一个latent，然后解码这个latent变成Metaloss网络参数，再通过metaloss网络更新z -> z^prime
#         super().__init__()
# #         self.phi = nn.Sequential(
# #             nn.Linear(input_len, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 1)
# #         )
#         self.input_len = input_len # z的长度
#         self.hidden_dim = hidden_dim # meta loss latent长度
#         # self.relu = mod_relu(self.input_len)
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#          )
        
#         self.latent = None
#         # self.latent.requires_grad_()
        
#         # 生成Metaloss网络最后一层参数
#         # self.param_generator = nn.Sequential(
#         #     nn.Linear(self.hidden_dim, self.hidden_dim * 32, bias=True),
#         #     nn.ReLU(),
#         #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4, bias=True),
#         #     # nn.ReLU(),
#         #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8, bias=True),
#         #     # nn.ReLU(),
#         #     nn.Linear(self.hidden_dim * 32, (self.input_len + 1) * self.hidden_dim, bias=True)
#         # )
        
#     def forward(self, X):
#         loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
#         # loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
#         # loss = torch.square(self.phi(X))
#         return loss


class MetaLoss(nn.Module):
    def __init__(self, input_len, hidden_dim, device):
        # meta loss和extrapolation_network一样，都是通过一个LSTM生成一个latent，然后解码这个latent变成Metaloss网络参数，再通过metaloss网络更新z -> z^prime
        super().__init__()
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        self.input_len = input_len # z的长度
        self.hidden_dim = hidden_dim # meta loss latent长度
        # self.relu = mod_relu(self.input_len)
        self.phi = nn.Sequential(
            nn.Linear(input_len, hidden_dim)
        ).to(torch.cfloat)
        
        self.latent = None
        # self.latent.requires_grad_()
        
        # 生成Metaloss网络最后一层参数
        # self.param_generator = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim * 32, bias=True),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4, bias=True),
        #     # nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8, bias=True),
        #     # nn.ReLU(),
        #     nn.Linear(self.hidden_dim * 32, (self.input_len + 1) * self.hidden_dim, bias=True)
        # )
        
    def forward(self, X):
        loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
        # loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
        # loss = torch.square(self.phi(X))
        return loss
    
    
class LatentMetaLoss(nn.Module):
    def __init__(self, input_len, hidden_dim, device, layers=1):
        # meta loss和extrapolation_network一样，都是通过一个LSTM生成一个latent，然后解码这个latent变成Metaloss网络参数，再通过metaloss网络更新z -> z^prime
        super().__init__()
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        self.input_len = input_len # z的长度
        self.hidden_dim = hidden_dim # meta loss latent长度
        # self.relu = mod_relu(self.input_len)
#         self.phi = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
        module_lst = [nn.Linear(input_len, hidden_dim)]
        for i in range(1, layers):
            module_lst.append(nn.ReLU())
            module_lst.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.phi = nn.Sequential(*module_lst)
        # self.latent.requires_grad_()
        
        # 生成Metaloss网络最后一层参数
        # self.param_generator = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim * 32, bias=True),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4, bias=True),
        #     # nn.ReLU(),
        #     # nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8, bias=True),
        #     # nn.ReLU(),
        #     nn.Linear(self.hidden_dim * 32, (self.input_len + 1) * self.hidden_dim, bias=True)
        # )
        
    def forward(self, X):
        loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
        # loss = torch.square(torch.norm(self.phi(X), p=2, dim=-1, keepdim=True))
        # loss = torch.square(self.phi(X))
        return loss


class MetaTS2Net(nn.Module):
    """
    Latent Optimization and Extrapolation
    """
    def __init__(self, target_model, configs):
        super(MetaTS2Net, self).__init__()
        # self.device = configs.device
        configs = SimpleNamespace(**configs)
        # basic settings
        # self.num_train_samples_per_episode = configs.num_train_samples_per_episode
        # self.num_test_samples_per_episode = configs.num_test_samples_per_episode
        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
        # self.channels = configs.enc_in
        # self.hidden_dim = configs.hidden_dim # z
        # self.channel_independent = configs.channel_independent

        # latent optimization and extrapolation settings 
        self.num_episode = configs.num_episode # latent外推LSTM输入长度
        self.latent_dim = configs.latent_dim  # 预测模型latent长度
        self.inner_lr = configs.inner_lr 
        self.inner_step = configs.inner_step
        self.device = configs.device
        self.inner_criterion = nn.MSELoss()
        self.target_model = target_model
        self.target_model.set_mode('finetune')
        self.smoothing_factor = configs.smoothing_factor
        self.enc_in = self.target_model.enc_in
        self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)
        self.mask_percentage = configs.mask_percentage
        self.beta_meta_loss = configs.beta_meta_loss
        self.beta_sim_loss = configs.beta_sim_loss
        self.latent_wd = configs.latent_wd
        self.sim_loss_dim = configs.sim_loss_dim
        self.sim_loss_score_num = configs.sim_loss_score_num
        self.sim_loss_use_alpha = configs.sim_loss_use_alpha
        self.meta_loss_dim = configs.meta_loss_dim
        self.latent_meta_loss_dim = configs.latent_meta_loss_dim
        self.latent_meta_loss_layers = configs.latent_meta_loss_layers
        self.extrapolation_dim = configs.extrapolation_dim
        self.extrapolation_linear_dim = configs.extrapolation_linear_dim
        self.adapt_latent_lr = configs.adapt_latent_lr
        self.adapt_sample_lr = configs.adapt_sample_lr
        self.beta_latent_meta_loss = configs.beta_latent_meta_loss
        self.beta_latent_sim_loss = configs.beta_latent_sim_loss
    


        self.voting_coef = nn.Parameter(torch.ones_like(self.target_model.latent) * 0)
        self.queue_latents = [self.target_model.latent.detach()] * self.num_episode # self.num_episode * self.latent_dim * self.enc_in
        # 生成预测模型latent
        # self.extrapolation_network = ExtrapolationNetwork(enc_in=self.latent_dim, hidden_dim=self.latent_dim * 8, linear_hidden_dim=self.latent_dim * 6)
        self.extrapolation_network = nn.ModuleList()
        # self.ema = nn.ModuleList()
        for i in range(self.enc_in):
            self.extrapolation_network.append(ExtrapolationNetwork(enc_in=self.latent_dim, hidden_dim=self.extrapolation_dim, linear_hidden_dim=self.extrapolation_linear_dim))
            # self.ema.append(MultiHeadEMA(self.latent_dim))
            # self.extrapolation_network.append(MultiHeadEMA(self.latent_dim))
        # 生成metaloss latent
        # self.sample_adaptation_network = LSTMEncoder(self.adapt_hidden, self.adapt_hidden * 4, self.adapt_hidden * 2, self.adapt_hidden)
        
        # metaloss网络
        # self.meta_loss = MetaLoss(self.target_model.hidden_dim + self.latent_dim, self.adapt_hidden * 4, self.device)
        # self.meta_loss = MetaLoss(self.z_dim + self.latent_dim + self.target_model.pred_len * self.target_model.enc_out, 2048, self.device)
        # self.meta_loss = MetaLoss(self.target_model.seq_len * self.target_model.enc_in + self.latent_dim + self.target_model.pred_len * self.target_model.enc_out, self.latent_dim * 16, self.device)
        # self.meta_loss = MetaLoss(self.target_model.seq_len * self.target_model.enc_in + self.latent_dim, 1024, self.device)
        # self.meta_loss = MetaLoss((self.target_model.seq_len * self.target_model.enc_in) // 2 + 1 + self.target_model.pred_len * self.target_model.enc_out, self.latent_dim, self.device)
        # self.meta_loss = MetaLoss(self.target_model.seq_len * self.target_model.enc_in, self.latent_dim, self.device)
        self.fft_half_n = self.target_model.seq_len // 2 + 1
        
        self.meta_loss = nn.ModuleList()
        self.sim_loss = nn.ModuleList()
        self.latent_meta_loss = nn.ModuleList()
        for i in range(self.enc_in):
            #self.meta_loss.append(MetaLoss(self.fft_half_n + self.latent_dim + self.target_model.pred_len, self.latent_dim * 8, self.device))
            # self.meta_loss.append(MetaLoss(self.target_model.seq_len, self.latent_dim * 4, self.latent_dim, self.device))
            self.sim_loss.append(SimLoss(self.target_model.seq_len, self.sim_loss_dim, self.latent_dim, self.device, num_sim=self.sim_loss_score_num, use_alpha=self.sim_loss_use_alpha))
            self.meta_loss.append(MetaLoss(self.fft_half_n + self.target_model.pred_len + self.latent_dim, self.meta_loss_dim, self.device))
            self.latent_meta_loss.append(LatentMetaLoss(self.target_model.seq_len + self.target_model.pred_len + self.latent_dim, self.latent_meta_loss_dim, self.device, self.latent_meta_loss_layers))
            

        # self.meta_loss = MetaLoss(self.fft_half_n + self.latent_dim + self.target_model.pred_len, self.latent_dim * 16, self.device)
        # self.meta_loss_mask = MetaLoss(self.fft_half_n * 2 + self.latent_dim + self.target_model.pred_len * self.target_model.enc_out, self.latent_dim * 16, self.device)
        # self.meta_loss = MetaLoss(self.fft_half_n + self.fft_half_n, self.latent_dim * 16, self.device)
        # 保存过去metaloss最好的latent，命名可能有些粗糙
        # self.queue_hidden = [self.meta_loss.latent.detach()] * self.num_episode
        # self.series_decomp = series_decomp(25)
        # self.memory_latent = Memory(self.memory_size, self.device)
        # self.tta_coef_network = nn.Sequential( # meta-tta emsembler
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.num_tta_task),
        #     nn.Softmax()
        # )
        self.running_latent_mean = torch.stack(self.queue_latents, dim=0).mean(dim=0)



#     def forward(self, X):
#         input_latents = torch.cat(self.queue_latents[-self.num_episode:], -1)
#         out_latent = self.extrapolation_network(input_latents)
#         self.target_model.latent = out_latent
#         y_hat, feat = self.target_model(X)
        
#         return y_hat
    
    def forward(self, X, adapt=False, sample_adapt=True, forward_latent=True, skip_connection=True):
        """ 
        X: D 里面的 X: (N, L, D)
        adapt: 如果False的话值输出预测，True输出预测以及X_test的hidden representation z
        input_z: 如果有的话就直接 target_model.forward_z(input_z), 不然的话就还要 z = self.target_model(X), z -> z^prime, target_model.forward_z(z)
        sample_sample：是否用metaloss adapt z -> z^prime
        """
        # Extrapolation stage, 预测出下一阶段的预测模型latent，把它放到target_model.latent
        if forward_latent:
            self.forward_latent(skip_connection)
        # Sample Adaptation Stage, 预测出下一阶段的Metaloss latent, 把它放到metaloss.latent
        # self.forward_hidden()
        # 得到预测值和adapt过后的z
        y_hat, z = self.forward_sample(X, sample_adapt)
        if adapt:
            return y_hat, z
        else:
            return y_hat

    
    def forward_sample(self, X, sample_adapt=True):
        # 不通过metaloss adapt
       #  _, z = self.target_model(X)
        X = self.revin_layer(X, 'norm')
        sample_latents = None
        if sample_adapt:
            # 通过metaloss adapt, 得到 z^prime
            X, sample_latents = self.sample_adapt(X)
        
        y_hat, z = self.target_model(X, sample_latents)
        y_hat = self.revin_layer(y_hat, 'denorm')
        
        return y_hat, z

    def forward_latent(self, skip_connection=True):
        # 变成 C * num_episode * latent_dim
        input_latents = torch.stack(self.queue_latents[-self.num_episode:], dim=0).permute(1, 0, 2)
        # input_latents = torch.cat(self.queue_latents[-self.num_episode:], dim=0).reshape(1, self.num_episode * self.latent_dim)
        out_latent = torch.ones_like(self.target_model.latent)
        for i in range(self.enc_in):
            out_latent[i, :] = self.extrapolation_network[i](input_latents[i:i+1, ...])

        if skip_connection:
            w = nn.functional.sigmoid(self.voting_coef)
            out_latent = w * out_latent + (1 - w) * input_latents.mean(dim=1)

        self.target_model.latent = out_latent
        

    def adapt(self, X_train, y_train, out=False, skip_connection=True):
        """
        X_train: (N, L, D)
        y_train: (N, H, D)
        out: 如果True输出meta的latent和target_mode的latent，False不输出只append到queue。
        """
        
        # 当我们有真实值了，将预测模型latent更新到最优
        # get latest latent
        self.forward_latent(skip_connection)
        latent_code = self.target_model.latent.detach()
        latent_code.requires_grad = True
        self.target_model.latent = latent_code
        adam = torch.optim.Adam([latent_code], lr=self.inner_lr, weight_decay=self.latent_wd)

        for i in range(self.inner_step):
            # self.meta_loss.latent = self.target_model.latent
            outputs, _ = self.forward_sample(X_train, sample_adapt=True)
            inner_loss = self.inner_criterion(outputs, y_train)
            adam.zero_grad()
            inner_loss.backward(inputs=[latent_code])
            adam.step()            
            if i % 10 == 0:
                print(f'Extrapolation adaptation: {inner_loss} ...')
            
        last_latent = self.target_model.latent.detach()
        
        if out:
            return last_latent
        else:
            self.queue_latents.append(last_latent)
            # self.running_latent_mean = self.smoothing_factor * last_latent + (1 - self.smoothing_factor) * self.running_latent_mean
            # self.queue_hidden.append(last_hidden)
    

    def sample_adapt(self, z):
        # SSL, sample adaptation
        N, L, C = z.shape # shape N, L, C
        unadapted_pred = self.target_model.forward(z)[0].detach().permute(0, 2, 1)
        z = z.detach()
        z = z.permute(0, 2, 1) # reshape to N, C, L for channel independence
        # z.requires_grad = True
        fft_output = torch.fft.rfft(z, dim=-1)
        mask = torch.ones_like(fft_output)
        mask[..., int(self.fft_half_n * (1 - self.mask_percentage)):] = 0
        # output = torch.ones_like(fft_output)
        if self.adapt_latent_lr == 0:
            output_latent = None
        else:
            output_latent = torch.zeros(N, C, self.latent_dim).to(z.device)

        output_x = torch.ones_like(fft_output)
        for i in range(1):
            for j in range(self.enc_in):
                curr_input = fft_output[:, j, :]
                curr_input.requires_grad = True
                curr_latent = self.target_model.latent[j, :].repeat(N, 1)
                
                metaloss_input = torch.cat([curr_input, curr_latent, unadapted_pred[:, j, :]], dim=-1)
                sim_loss = self.sim_loss[j](torch.fft.irfft(curr_input, dim=-1), curr_latent)
                meta_loss = self.meta_loss[j](metaloss_input)
                total_loss = self.beta_meta_loss * meta_loss + self.beta_sim_loss * sim_loss
                # loss = self.meta_loss[j](a, curr_latent)
                grad_x = torch.autograd.grad(total_loss, curr_input, torch.ones_like(total_loss), create_graph=True, retain_graph=True)[0]
                # grad_latent = torch.autograd.grad(latent_loss, curr_latent, torch.ones_like(latent_loss), create_graph=True, retain_graph=True)[0]
                
                if self.adapt_latent_lr != 0:
                    latent_loss = self.beta_latent_meta_loss * self.latent_meta_loss[j](torch.cat([z[:, j, :], curr_latent, unadapted_pred[:, j, :]], dim=-1)) + self.beta_latent_sim_loss * sim_loss
                   
                    grad_latent = torch.autograd.grad(latent_loss, curr_latent, torch.ones_like(latent_loss), create_graph=True, retain_graph=True)[0]
                    output_latent[:, j, :] = curr_latent - self.adapt_latent_lr * grad_latent
                
                output_x[:, j, :] = curr_input - self.adapt_sample_lr * grad_x * mask[:, j, :]
        
        # output = mask * output_x
        output = output_x
        output = torch.fft.irfft(output, dim=-1)
        output = output.permute(0, 2, 1)
        
        return output, output_latent
    

    def predict(self, X):
        predictions = self.forward(X)

        return predictions

    def _init_tensor(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor
    
    def set_mode(self, mode):
        self.target_model.set_mode(mode)
       # self.meta_loss.set_mode(mode)
        if mode == 'warmup':
            for p in self.extrapolation_network.parameters():
                p.requires_grad = False
                
        else:
            for p in self.extrapolation_network.parameters():
                p.requires_grad = True

            
            # for p in self.sample_adaptation_network.parameters():
            #     p.requires_grad = True
                
            # for p in self.meta_loss.parameters():
            #     p.requires_grad = True
            

    def _init_tensor(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor

    def fetch_latent(self, latent):
        # num_episodes, C, latent_dim
        full_latents = torch.stack(self.queue_latents, dim=0)
        dis = nn.functional.cosine_similarity(latent.detach().repeat(len(full_latents), 1, 1), full_latents, dim=-1)
        dis, idx = dis.topk(self.num_episode, dim=0)
        full_latents = full_latents.gather(0, idx.unsqueeze(-1).expand(-1, -1, self.latent_dim))
        att = nn.functional.softmax(dis, dim=0)
        out_latents = torch.einsum('ijk, ij -> jk', [full_latents, att])
        
        return out_latents
    
    def generate_latent_samples(self):
        q_len = len(self.queue_latents)
        latent_X = []
        latent_y = []
        if q_len > self.num_episode * 2:
            for i in range(self.num_episode, q_len - self.num_episode):
                curr_latent_X = self.queue_latents[i:i+self.num_episode]
                curr_latent_y = self.queue_latents[i+self.num_episode]
                curr_latent_X = torch.stack(curr_latent_X, dim=0).permute(1, 0, 2)
                latent_X.append(curr_latent_X)
                latent_y.append(curr_latent_y)
            
            latent_X = torch.stack(latent_X, dim=0)
            latent_y = torch.stack(latent_y, dim=0)
            
            return latent_X, latent_y
        else:
            return None, None
                