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
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
import logging



# Bus dataset
class BusLoadDataset(Dataset):
    def __init__(self, 
         data_path, 
         seq_len, 
         device, 
         warm_up_time=None,
         pred_len=1, 
         target_col='load',
         mode='MS',
         dt_col='time'):
        
        self.X = None
        self.y = None
        self.curr_X = None
        self.curr_y = None
        self.data_path = data_path
        self.seq_len = seq_len
        self.device = device
        self.warm_up_time = warm_up_time
        self.pred_len = pred_len
        self.flag = -1 if (mode == 'MS') or (mode == 'S') else 0
        self.target_col = target_col
        self.mode = mode
        self.dt_col = dt_col
        self._read_data()
    
    def _read_data(self):
        assert self.mode in ['MS', 'S', 'M']
        data = pd.read_csv(self.data_path).ffill().bfill()
        
        if self.dt_col:
            data[self.dt_col] = pd.to_datetime(data[self.dt_col])
            data.set_index(self.dt_col, inplace=True)

        # move target to the last dimension
        if (self.mode == 'S') or (self.mode == 'MS'):
            column_names = list(data.columns)
            column_names.remove(self.target_col)
            column_names = column_names + [self.target_col]
            data = data[column_names]
        
        if self.mode == 'S':
            data = pd.DataFrame(data[self.target_col])
            
        self.X = data.values

        if self.mode == 'MS':
            self.y = pd.DataFrame(data[self.target_col]).values
        else:
            self.y = data.values
        
        if self.warm_up_time:
            # standardize using warm_up
            start, end = self.warm_up_time
            warm_up_X = self.X[start:end]
            warm_up_y = self.y[start:end]
            sc_X = sklearn_StandardScaler()
            sc_y = sklearn_StandardScaler()
            sc_X.fit(warm_up_X)
            sc_y.fit(warm_up_y)
            self.X = sc_X.transform(self.X)
            self.y = sc_y.transform(self.y)
        
        self.X = self.to_device(self.X)
        self.y = self.to_device(self.y)
    
    def __getitem__(self, idx):
        x_train = self.curr_X[idx:idx + self.seq_len, 0:]
        y_train = self.curr_y[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.flag:]

        return x_train.float(), y_train.float()

    def __len__(self):
        return len(self.curr_X) - self.seq_len - self.pred_len + 1

    def update_dataset(self, start_end_time):
        start, end = start_end_time
        self.curr_X = self.X[start:end]
        self.curr_y = self.y[start:end]
    
    def to_device(self, arr):
        return torch.from_numpy(arr).to(self.device)

# ECL dataset
class ECLDataset(Dataset):
    def __init__(self, 
         data_path, 
         seq_len, 
         device, 
         warm_up_time=None,
         pred_len=1, 
         col_num=20,
         target_col='load',
         mode='MS',
         dt_col='time'):
        
        self.X = None
        self.y = None
        self.curr_X = None
        self.curr_y = None
        self.data_path = data_path
        self.seq_len = seq_len
        self.device = device
        self.warm_up_time = warm_up_time
        self.pred_len = pred_len
        self.flag = -1 if (mode == 'MS') or (mode == 'S') else 0
        self.target_col = target_col
        self.mode = mode
        self.dt_col = dt_col
        self.col_num = col_num
        self._read_data()
    
    def _read_data(self):
        assert self.mode in ['MS', 'S', 'M']
        data = pd.read_csv(self.data_path).ffill().bfill()
        
        if self.dt_col:
            data[self.dt_col] = pd.to_datetime(data[self.dt_col])
            data.set_index(self.dt_col, inplace=True)

        # move target to the last dimension
        if (self.mode == 'S') or (self.mode == 'MS'):
            column_names = list(data.columns)
            column_names.remove(self.target_col)
            column_names = column_names + [self.target_col]
            data = data[column_names]
        
        if self.mode == 'S':
            data = pd.DataFrame(data[self.target_col])

        # random select columns
        selected_columns = np.random.choice(list(data.columns), self.col_num, replace=False)
        logging.info('COLUMNS SELECTED FOR ECL ARE: {}'.format(selected_columns))
        data = data[selected_columns]
            
        self.X = data.values

        if self.mode == 'MS':
            self.y = pd.DataFrame(data[self.target_col]).values
        else:
            self.y = data.values
        
        if self.warm_up_time:
            # standardize using warm_up
            start, end = self.warm_up_time
            warm_up_X = self.X[start:end]
            warm_up_y = self.y[start:end]
            sc_X = sklearn_StandardScaler()
            sc_y = sklearn_StandardScaler()
            sc_X.fit(warm_up_X)
            sc_y.fit(warm_up_y)
            self.X = sc_X.transform(self.X)
            self.y = sc_y.transform(self.y)
        
        self.X = self.to_device(self.X)
        self.y = self.to_device(self.y)
    
    def __getitem__(self, idx):
        x_train = self.curr_X[idx:idx + self.seq_len, 0:]
        y_train = self.curr_y[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.flag:]

        return x_train.float(), y_train.float()

    def __len__(self):
        return len(self.curr_X) - self.seq_len - self.pred_len + 1

    def update_dataset(self, start_end_time):
        start, end = start_end_time
        self.curr_X = self.X[start:end]
        self.curr_y = self.y[start:end]
    
    def to_device(self, arr):
        return torch.from_numpy(arr).to(self.device)