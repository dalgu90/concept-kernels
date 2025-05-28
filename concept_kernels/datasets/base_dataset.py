from typing import Union
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import xgboost as xgb

from concept_kernels.utils.graph_utils import get_topk_eigen


class BaseDataset(Dataset):
    def __init__(
        self,
        X_num: Union[torch.Tensor, np.array, None],
        X_cat: Union[torch.Tensor, np.array, None],
        y: Union[torch.Tensor, np.array],
        split: str = "all",
        kg: Union[Data, None] = None,
        metadata: dict = None,
        is_preprocessed: bool = False,
    ):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y
        self.split = split
        self.kg = kg
        self.metadata = metadata
        self.dmatrix = None
        self.is_preprocessed = is_preprocessed

    def __len__(self):
        return len(self.X_num) if self.X_num is not None else len(self.X_cat)

    def __getitem__(self, idx):
        if not self.is_preprocessed:
            class_name = {self.__class__.__name__}
            warnings.warn(f"{class_name}.__getitem__ is called before preprocessing")

        # X_num and X_cat can be None, torch.Tensor, or np.ndarray
        X_num, X_cat = None, None
        if self.X_num is not None:
            if isinstance(self.X_num, np.ndarray):
                self.X_num = torch.tensor(self.X_num, dtype=torch.float32)
            X_num = self.X_num[idx]
        if self.X_cat is not None:
            if isinstance(self.X_cat, np.ndarray):
                self.X_cat = torch.tensor(self.X_cat, dtype=torch.float32)
            X_cat = self.X_cat[idx]
        # y can be torch.Tensor or np.ndarray
        if isinstance(self.y, torch.Tensor):
            y = self.y[idx]
        else:
            y = torch.tensor(self.y[idx])
        ret = {
            'X_num': X_num,
            'X_cat': X_cat,
            'label': y,
        }
        return ret

    def collate_fn(self, examples):
        # X_cat and X_num can be None, but not both at the same time
        X_cat, X_num = None, None
        if examples[0]['X_num'] is not None:
            X_num = torch.stack([e['X_num'] for e in examples])
        if examples[0]['X_cat'] is not None:
            X_cat = torch.stack([e['X_cat'] for e in examples])
        ret = {
            'X_num': X_num,
            'X_cat': X_cat,
            'label': torch.stack([e['label'] for e in examples])
        }
        return ret

    def to_dmatrix(self):
        if not self.is_preprocessed:
            class_name = {self.__class__.__name__}
            warnings.warn(f"{class_name}.to_dmatrix is called before preprocessing")

        X = []
        if self.X_num is not None:
            if isinstance(self.X_num, np.ndarray):
                self.X_num = torch.tensor(self.X_num, dtype=torch.float32)
            X.append(self.X_num)
        if self.X_cat is not None:
            if isinstance(self.X_cat, np.ndarray):
                self.X_cat = torch.tensor(self.X_cat, dtype=torch.float32)
            X.append(self.X_cat)
        X = torch.cat(X, dim=0)
        y = torch.tensor(self.y)
        if self.dmatrix is None:
            self.dmatrix = xgb.DMatrix(X, label=y)
        return self.dmatrix
