import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)

def seed(value=42, deterministic=True):
    logger.debug(f"Setting seed to {value}, deterministic={deterministic}")
    os.environ['PYTHONHASHSEED']=str(value)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

def show_trainable_params(model):
    total_params = 0
    logger.info("Trainable Parameters")
    logger.info(f"{'(Name)':40} {'(Shape)':20} (Count)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name:40} {str(list(param.shape)):20} {param.numel()}")
            total_params += param.numel()
    logger.info(f"=> Total Trainable Parameters: {total_params}")

def show_trainable_params_xgb(model):
    if hasattr(model, 'get_total_params'):
        total_params = model.get_total_params()
    else:
        tree_dump = model.get_booster().get_dump()
        splits = sum(line.count('[') for tree in tree_dump for line in tree.split('\n'))
        leaves = sum(line.count('leaf=') for tree in tree_dump for line in tree.split('\n'))
        total_params = splits * 2 + leaves
    logger.info(f"=> Total Trainable Parameters: {total_params}")
