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
import logging

# ONLINE LEARNING ENVs
def data_generator(datasets, split_times):
    if len(datasets) == 3:
        train, valid, test = datasets
        for train_s, train_e, valid_s, valid_e, test_s, test_e in split_times:
            train.update_dataset((train_s, train_e))
            valid.update_dataset((valid_s, valid_e))
            test.update_dataset((test_s, test_e))

            yield train, valid, test
    else:
        train, test = datasets
        for train_s, train_e, test_s, test_e in split_times:
            train.update_dataset((train_s, train_e))
            test.update_dataset((test_s, test_e))

            yield train, test
            
            
class OnlineEnv:
    """
    Online Learning Environment
    
    >>> train = BusLoadDataset('datasets/longmen_2.csv', 7 * 96, device, model='LSTM', mode='train', look_ahead=5)
    >>> test = BusLoadDataset('datasets/longmen_2.csv', 7 * 96, device, model='LSTM', mode='test', look_ahead=5)
    >>> datasets = (train, test)
    >>> split_times = [(0, 1000, 1000, 2000)] # (i.e., [(train_start, train_end, test_start, test_end)])
    >>> metrics = {'mse': mean_squared_error, 'mae': mean_absolute_error}
    >>> online_env = OnlineEnv(datasets, split_times, metrics)
    >>> warmup_X_train_ds, curr_X_test_tensor = online_env.reset() # starting, get warmup, train on (warmup_X_train_ds, curr_X_test_tensor), and make predictions.
    >>> predictions = model.predict(curr_X_test_tensor)
    >>> next_X_train_ds, curr_y_test_tensor, performance_dict, curr_X_test_tensor = online_env.step(predictions)
    """
    def __init__(self, datasets, split_times, metrics):
        self._train, self._test = datasets
        self.split_times = split_times
        self.metrics = metrics
        self._curr_y_test = None
        self._curr_X_test = None
        self._curr_test_dl = None
        self._curr_split_time = None
        self._performance_dict = {}
        
    def step(self, pred):
        if self._curr_split_time is None:
            logging.warning('please reset before step !')
        else:
            y_test = self._curr_y_test
            performance_dict = {}
            
            for name, metric in self.metrics.items():
                error = metric(pred, y_test)
                performance_dict[name] = error.cpu().detach()
                logging.info('The test {} is {} ...'.format(name, error))
                
            self._update(performance_dict)
            
            return (self._train, y_test, performance_dict, self._curr_X_test)
        
    def reset(self, reset_time=0):
        # reset the online Environment, return first (train_dl, test_dl)
        self._curr_split_time = reset_time
        self._update()
    
        return self._train, self._curr_X_test
    
    def _update(self, performance_dict=None):
        if self._curr_split_time < len(self.split_times):
            train_s, train_e, test_s, test_e = self.split_times[self._curr_split_time]
            self._train.update_dataset((train_s, train_e))
            self._test.update_dataset((test_s, test_e))
            self._curr_test_dl = DataLoader(self._test, batch_size=len(self._test))
            self._curr_X_test, self._curr_y_test = next(iter(self._curr_test_dl))
            self._curr_split_time = self._curr_split_time + 1
        else:
            self._curr_split_time = None
        
        # update performance dict
        if performance_dict:
            for k, v in performance_dict.items():
                if k not in self._performance_dict.keys():
                    self._performance_dict[k] = [v]
                else:
                    self._performance_dict[k].append(v)
                    
    def is_done(self):
        if self._curr_split_time is not None:
            return False
        else:
            logging.info('\n Online Learning is finished: \n')
            for k, v in self._performance_dict.items():
                logging.info(f'The average {k} is: {np.mean(v)}')
                
            return True
        
    def get_y_test(self):
        return self._curr_y_test
    
    def get_curr_split_time(self):
        if self._curr_split_time > 0:
            return self.split_times[self._curr_split_time - 1]
        else:
            return self._curr_split_time
    

class OnlineEnvModel:
    """
    Naive baseline模型用这个
    """
    def __init__(self, configs):
        self.configs = configs
        self.model = configs['model']
    
    def fit(self, train_dl, test_dl, env, epochs=None, warm_up=False, **kwargs):
        optim = self.configs['optim'][0](self.model.parameters(), **self.configs['optim'][1])
        loss_fun = self.configs['loss'] 
        metrics = self.configs['metrics']
        if warm_up:
            for i in range(1, self.configs['warm_up_epochs'] + 1):
                self.model.train()
                total_loss = []
                for X_train, y_train in train_dl:
                    y_pred = self.model(X_train)
                    loss = loss_fun(y_pred, y_train)
                    total_loss.append(loss.item())

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                logging.info('Epoch {}: The training loss is {} ...'.format(i, np.mean(total_loss)))
        else:
            optim = self.configs['optim'][0](self.model.fine_tune_layer.parameters(), **self.configs['optim'][1]) 
            for i in range(1, self.configs['num_epochs'] + 1):
                self.model.train()
                total_loss = []
                for X_train, y_train in train_dl:
                    y_pred = self.model(X_train)
                    loss = loss_fun(y_pred, y_train)
                    total_loss.append(loss.item())

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                logging.info('Epoch {}: The training loss is {} ...'.format(i, np.mean(total_loss)))
            
#             self.model.eval()
#             y_pred = self.model.predict(self.standardize(X_test))
#             for name, metric in metrics.items():
#                 error = metric(y_pred, y_test)

#                 print('The validation {} is {} ...'.format(name, error))
    
    def predict(self, X):
        self.model.eval()
        return self.model.predict(X)
    
    def update_model(self, model):
        self.model = model

        
class OnlineMeta2Model:
    """
    Meta2Net模型用这个
    """
    def __init__(self, configs):
        self.configs = configs
        self.model = configs['model']
    
    def fit(self, train_dl, test_dl, env, epochs=None, warm_up=False, **kwargs):
        loss_fun = self.configs['loss'] 
        metrics = self.configs['metrics']
        warm_up_mode = self.configs.get('warm_up_mode')
        if warm_up:
            # 是否为seperate warm_up
            if warm_up_mode:
                self.model.set_mode('warmup')
                # optim = self.configs['warm_up_optim'][0](list(self.model.parameters()) + list(self.model.meta_loss.parameters()), **self.configs['warm_up_optim'][1])
                optim = self.configs['warm_up_optim'][0](self.model.parameters(), **self.configs['warm_up_optim'][1])
                warm_up_epochs = self.configs['warm_up_epochs']
                # warm up target_model
                for i in range(1, warm_up_epochs + 1):
                    self.model.train()
                    total_loss = []
                    for X_train, y_train in train_dl:
                        y_pred = self.model(X_train, sample_adapt=False)
                        loss = loss_fun(y_pred, y_train)
                        total_loss.append(loss.item())

                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    
                    logging.info('Epoch {}: The Warm up training loss is {} ...'.format(i, np.mean(total_loss)))   
                
                optim = self.configs['warm_up_optim'][0](list(self.model.meta_loss.parameters()) + list(self.model.sim_loss.parameters()), **self.configs['warm_up_optim'][1])
                
                # warm up metaloss
                for i in range(1, warm_up_epochs + 1):
                    self.model.train()
                    total_loss = []
                    for X_train, y_train in train_dl:
                        y_pred = self.model(X_train, sample_adapt=True)
                        loss = loss_fun(y_pred, y_train)
                        total_loss.append(loss.item())

                        optim.zero_grad()
                        loss.backward(inputs=list(self.model.meta_loss.parameters()))
                        optim.step()
                    
                    logging.info('Epoch {}: The Warm up training loss for metaloss is {} ...'.format(i, np.mean(total_loss)))   
            else:
                # 一起warm up
                self.model.set_mode('warmup')
                # optim = self.configs['warm_up_optim'][0](list(self.model.parameters()) + list(self.model.meta_loss.parameters()), **self.configs['warm_up_optim'][1])
                optim = self.configs['warm_up_optim'][0](self.model.parameters(), **self.configs['warm_up_optim'][1])
                warm_up_epochs = self.configs['warm_up_epochs']
                # warm up target_model
                for i in range(1, warm_up_epochs + 1):
                    self.model.train()
                    total_loss = []
                    for X_train, y_train in train_dl:
                        y_pred = self.model(X_train, sample_adapt=True)
                        loss = loss_fun(y_pred, y_train)
                        total_loss.append(loss.item())

                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    
                    logging.info('Epoch {}: The Warm up training loss is {} ...'.format(i, np.mean(total_loss)))   

            
        else:
            self.model.set_mode('finetune')
            self.model.train()
            X_train, y_train = next(iter(train_dl))
            self.model.adapt(X_train, y_train)
    
    def predict(self, X):
        self.model.eval()
        return self.model.predict(X)
    

    def update_meta_model(self, X_test, y_test, epochs):
        self.model.train()
        loss_fun = self.configs['loss']
        tune_parameters = list(self.model.extrapolation_network.parameters()) + list(self.model.meta_loss.parameters()) + list(self.model.sim_loss.parameters()) + list(self.model.target_model.param_generator.parameters()) + [self.model.voting_coef] + list(self.model.latent_meta_loss.parameters())
        # tune_parameters = list(self.model.extrapolation_network.parameters()) + list(self.model.meta_loss.parameters()) + list(self.model.sim_loss.parameters()) + [self.model.voting_coef]
        # tune_parameters = list(self.model.extrapolation_network.parameters())
        optim = self.configs['warm_up_optim'][0](tune_parameters, **self.configs['warm_up_optim'][1])
        # meta_loss_optim = self.configs['warm_up_optim'][0](self.model.sample_adaptation_network.parameters(), **self.configs['warm_up_optim'][1])
        # meta_loss_optim = self.configs['warm_up_optim'][0](self.model.meta_loss.parameters(), **self.configs['warm_up_optim'][1])
        reg_latent_coef = self.configs.get('reg_latent_coef')
        if reg_latent_coef > 0:
            last_latent = self.model.adapt(X_test, y_test, out=True, skip_connection=True)
        
        
        # then update extrapolation network
        for i in range(1, epochs + 1):
            # y_pred = self.model(X_test, input_z=z)
            y_pred = self.model(X_test)
            loss = loss_fun(y_pred, y_test) 
            if reg_latent_coef > 0:
                loss = reg_latent_coef * torch.square(torch.norm(self.model.target_model.latent - last_latent, p=2)) + loss
                
            # fetch examples from memory
            # if len(self.model.queue_latents) >= (self.model.num_episode + 1):
            #     input_latents = self.model.queue_latents[-self.model.num_episode - 1:-1]
            #     input_latents = torch.stack(input_latents, dim=0).permute(1, 0, 2)
            #     target_latent = self.model.queue_latents[-1]
            #     out_latent = self.model.forward_latent(input_latents=input_latents, output_latent=True)
            #     loss_der_latent = torch.square(torch.norm(out_latent -target_latent, p=2))
            #     loss = loss + loss_der_latent * 0.3
            
            optim.zero_grad()
            loss.backward(inputs=tune_parameters)
            optim.step()      
            
            logging.info('Epoch {}: The extrapolation loss on test is {} ...'.format(i, loss.item()))   
                
        # self.model.target_model.latent = start_latent
        # # first update MetaLoss
        # for i in range(1, epochs + 1):
        #     y_pred = self.model(X_test, forward_latent=False, sample_adapt=True)
        #     loss = loss_fun(y_pred, y_test)
        #     # loss = 0.1 * torch.norm(self.model.meta_loss.latent - last_hidden, p=2) + 0.9 * loss
        #     meta_loss_optim.zero_grad()
        #     loss.backward(inputs=list(self.model.meta_loss.parameters()))
        #     meta_loss_optim.step()      
            
        #     logging.info('Epoch {}: The Meta loss on test is {} ...'.format(i, loss.item()))   



class OnlineEnvTrain:
    """
    train_cls: 定义了哪种模式进行online learning，同时定义了如何进行和online_env的交互，记录每个阶段的预测值，真实值，metrics等
    """
    def __init__(self, env, train_step, configs):
        self.env = env
        self.configs = configs
        self.train_step = train_step(self.configs)
        self.pred_df = pd.DataFrame([])
        self.true_df = pd.DataFrame([])
        self.score_df = pd.DataFrame([])
        self.test_dls = []
        self.model_lst = []
    
    def naive(self):
        logging.info('Perform Naive online learning ...')
        train_ds, X_test = self.env.reset()
        self.reset()
        meta_train_length = self.configs['meta_train_length']

        i = 0
        while not self.env.is_done(): 
            if self.configs['batch_size'] == -1:
                batch_size = len(train_ds)
            else:
                batch_size = self.configs['batch_size']
                
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=self.configs['shuffle'])
            logging.info('\n current split time is {} \n'.format(self.env.get_curr_split_time()))
            if i == 0:
                self.train_step.fit(train_dl, X_test, self.env, warm_up=True)
                self.model_lst.append(self.train_step.model)
                
            y_pred = self.train_step.predict(X_test)

            train_ds, y_test, performance_dict, X_test = self.env.step(y_pred.detach())
            if i > meta_train_length:
                self.record(y_pred, y_test, performance_dict)            
                
            i += 1
                
    def retrain(self):
        logging.info('Perform Retraining ...')
        train_ds, X_test = self.env.reset()
        self.reset()
        meta_train_length = self.configs['meta_train_length']
        i = 0
        while not self.env.is_done(): 
            logging.info('\n current split time is {} \n'.format(self.env.get_curr_split_time()))
            self.train_step.update_model(deepcopy(self.configs['model']))
            
            tmp_dl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=self.configs['shuffle'])
            X_train, y_train = next(iter(tmp_dl))
            
            train_dl = DataLoader(train_ds, batch_size=self.configs['batch_size'], shuffle=self.configs['shuffle'])
            
            self.train_step.fit(train_dl, X_test, self.env)
            self.model_lst.append(deepcopy(self.train_step.model))
            y_pred = self.train_step.predict(X_test)
            train_ds, y_test, performance_dict, X_test = self.env.step(y_pred)
            if i > meta_train_length:
                self.record(y_pred, y_test, performance_dict)     
                
            i += 1
    
    def fine_tune(self):
        logging.info('perform fine tuning ...')
        self.reset()
        train_ds, X_test = self.env.reset()
        meta_train_length = self.configs['meta_train_length']
        i = 0
        while not self.env.is_done(): 
            logging.info('\n current split time is {} \n'.format(self.env.get_curr_split_time()))
            
            if i == 0:
                train_dl = DataLoader(train_ds, batch_size=self.configs['warm_up_batch_size'], shuffle=self.configs['shuffle'])
                self.train_step.fit(train_dl, X_test, self.env, warm_up=True)
            else:
                train_dl = DataLoader(train_ds, batch_size=self.configs['batch_size'], shuffle=self.configs['shuffle'])
                self.train_step.fit(train_dl, X_test, self.env)
                
            self.model_lst.append(self.train_step.model)
    
            y_pred = self.train_step.predict(X_test)
            train_ds, y_test, performance_dict, X_test = self.env.step(y_pred)
            if i > meta_train_length:
                self.record(y_pred, y_test, performance_dict)
            
            i += 1
                
    def meta_finetune(self):
        def inner_step(train_ds, X_test, batch_size, warm_up, update_meta=False, record=False, epochs=10):
            train_dl = DataLoader(deepcopy(train_ds), batch_size=batch_size, shuffle=self.configs['shuffle'])
            self.train_step.fit(train_dl=train_dl, test_dl=None, env=self.env, warm_up=warm_up)
            self.train_step.model.eval()
            y_pred = self.train_step.predict(X_test)
            prev_X_test = X_test
            train_ds, y_test, performance_dict, X_test = self.env.step(y_pred.detach())
            if record:
                self.record(y_pred.detach(), y_test, performance_dict)
                
            if update_meta:
                self.train_step.update_meta_model(prev_X_test, y_test, epochs)
            
            return train_ds, X_test
            
        logging.info(f'Perform Meta Finetuning ...')
        self.reset()
        train_ds, X_test = self.env.reset()
        meta_train_length = self.configs['meta_train_length']
        meta_train_epochs = self.configs['meta_train_epochs']
        # warm_up
        logging.info('Warm up start: warm up time is {}'.format(self.env.get_curr_split_time()))
        _, _ = inner_step(train_ds, X_test, self.configs['warm_up_batch_size'], warm_up=True, update_meta=False)
        
        # meta train
        for epoch in range(1, meta_train_epochs + 1):
            train_ds, X_test = self.env.reset(1)
            logging.info('Meta Train epoch {} starts:'.format(epoch))

            for meta_train_step in range(1, meta_train_length + 1):
                logging.info('\n current split time is {} \n'.format(self.env.get_curr_split_time()))
                train_ds, X_test = inner_step(train_ds, X_test, len(train_ds), False, True, False, epochs=self.configs['num_epochs'])
                
        # meta test
        train_ds, X_test = self.env.reset(1 + meta_train_length)
        logging.info('Performing Meta Testing ...')
        while not self.env.is_done():
            logging.info('\n current split time is {} \n'.format(self.env.get_curr_split_time()))
            train_ds, X_test = inner_step(train_ds, X_test, len(train_ds), False, True, True, epochs=self.configs['num_epochs_update'])
         

        test_after = self.train_step.model.sample_adapt(X_test)
        # # inject missing values

        # missing_X_test = X_test.detach()
        # missing_X_test[:, -10:, :] = 0

        import pickle as pkl

        with open('adapt.pkl', 'wb') as f:
            pkl.dump((X_test, test_after), f)
        
        with open('latents.pkl', 'wb') as f:
            pkl.dump(self.train_step.model.queue_latents, f)
        # with open('mask.pkl', 'wb') as f:
        #     pkl.dump(self.train_step.model.mask, f)

        # with open('missing_values.pkl', 'wb') as f:
        #     pkl.dump((missing_X_test, self.train_step.model.sample_adapt(missing_X_test)), f)
        
        



    def record(self, y_pred, y_test, performance_dict):
        # select load only
        y_pred = y_pred[:, :, -1].cpu().detach().numpy()
        y_test = y_test[:, :, -1].cpu().detach().numpy()
        self.pred_df = pd.concat([self.pred_df, pd.DataFrame(y_pred)], axis=0).reset_index(drop=True)
        self.true_df = pd.concat([self.true_df, pd.DataFrame(y_test)], axis=0).reset_index(drop=True)
        self.score_df = pd.concat([self.score_df, pd.DataFrame(performance_dict, index=[0])], axis=0).reset_index(drop=True)

    
    def reset(self):
        self.pred_df = pd.DataFrame([])
        self.true_df = pd.DataFrame([])
        self.score_df = pd.DataFrame([])
        self.test_dls = []
        self.model_lst = []