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
import random


def module_update(model):
    # replace model parameters with a_p
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            model._parameters[param_key] = p.update
            p.update = None
            
    # update buffer
    for buff_key in model._buffers:
        buff = model._buffers[buff_key]
        if buff is not None and buff.update is not None:
            model._buffers[buff_key] = buff.update
            buff.update = None
    
    # Recursion
    for module_key in model._modules:
        model._modules[module_key] = module_update(model._modules[module_key])
    
    return model


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def detach_module(module, keep_requires_grad=False):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Detaches all parameters/buffers of a previously cloned module from its computational graph.
    Note: detach works in-place, so it does not return a copy.
    **Arguments**
    * **module** (Module) - Module to be detached.
    * **keep_requires_grad** (bool) - By default, all parameters of the detached module will have
    `requires_grad` set to `False`. If this flag is set to `True`, then the `requires_grad` field
    will be the same as the pre-detached module.
    **Example**
    ~~~python
    net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone, keep_requires_grad=True)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            requires_grad = module._parameters[param_key].requires_grad
            detached = module._parameters[param_key].detach_()
            if keep_requires_grad and requires_grad:
                module._parameters[param_key].requires_grad_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
            if keep_requires_grad:  # requires_grad checked above
                module._buffers[buffer_key].requires_grad_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key], keep_requires_grad=keep_requires_grad)

class Memory:
    def __init__(self, memory_size, device):
        self.memory_size = memory_size
        self.deque = deque(maxlen=memory_size)
    
    def append_all(self, X):
        self.deque.append(X.detach())
        
    def append(self, X, N):
        for i in range(N):
            self.deque.append(X[i:i+1, ...].detach())
    
    def fetch(self):
        return torch.cat(list(self.deque), dim=0).to(device)

def gen_split_time(interval, warm_up, win, mode='retrain', algo=None):
    if algo == 'meta':
        out_interval = [(warm_up[0], warm_up[1], warm_up[1], warm_up[1] + win), (warm_up[0], warm_up[1], warm_up[1], warm_up[1] + win)]
    else:
        out_interval = [(warm_up[0], warm_up[1], warm_up[1], warm_up[1] + win)]
    
    if mode == 'fine_tune':
        for i in range(warm_up[1] + win, interval[-1], win):
            if i + win >= interval[-1]:
                valid_end = interval[-1]
            else:
                valid_end = i + win

            if i + win + win > interval[-1]:
                out_interval.append((out_interval[-1][-2], i, i, interval[-1]))
                break

            out_interval.append((out_interval[-1][-2], i, i, valid_end))
    else:
        for i in range(warm_up[1] + win, interval[-1], win):
            if i + win >= interval[-1]:
                valid_end = interval[-1]
            else:
                valid_end = i + win

            if i + win + win > interval[-1]:
                out_interval.append((warm_up[0], i, i, interval[-1]))
                break

            out_interval.append((warm_up[0], i, i, valid_end))
        
    return out_interval


def score_all(train_cls_lst, mode='retrain', repeat=1, seed=16, *args, **kwargs):
    out_train_cls_outer = {}
    out_mean_acc_dict = {}
    for train_cls, name in train_cls_lst:
        # out_train_cls = []
        acc_dict = {}
        logging.info('\n')
        logging.info('SCORING FOR DATASET {}'.format(name))
        logging.info('\n')
        for r in range(repeat):
            curr_train_cls = deepcopy(train_cls)
            getattr(curr_train_cls, mode)(*args, **kwargs)
            # out_train_cls.append(train_cls)
            score_dict = curr_train_cls.score_df.mean().to_dict()
            logging.info('current score is: {} ...'.format(score_dict))
            for score_name, score in score_dict.items():
                if score_name not in acc_dict:
                    acc_dict[score_name] = [score]
                else:
                    acc_dict[score_name].append(score)

            if name not in out_train_cls_outer:
                out_train_cls_outer[name] = [curr_train_cls]
            else:
                out_train_cls_outer[name].append(curr_train_cls)
            
            curr_train_cls.train_step.model.cpu()
            del curr_train_cls
            torch.cuda.empty_cache()
            
        out_mean_acc_dict[name] = {k: np.mean(v) for k, v in acc_dict.items()}
        logging.info('\n')
        logging.info(f' \n The Online Score (Meta Testing) For {name} is {out_mean_acc_dict[name]} \n')
        logging.info('\n')


    logging.info('\n')
    logging.info(out_mean_acc_dict)

    return out_train_cls_outer, out_mean_acc_dict
        

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(file_name, logger_name):
    """
    Set logging format and output logging to file.
    If file name is empty or None, logging will not be saved.

    Args:
        file_name:
        logger_name:

    Returns:

    """
    from imp import reload
    reload(logging)

    stream_handler = logging.StreamHandler()
    stream_handler_fmt = CustomFormatter()
    stream_handler.setFormatter(stream_handler_fmt)
    handlers = [stream_handler]

    if file_name:
        file_handler = logging.FileHandler(file_name, mode='a')
        file_handler_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_handler_fmt)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)

    return logging.getLogger(logger_name)