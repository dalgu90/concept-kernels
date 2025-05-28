from typing import Union

import torch
from torch_geometric.data import Data
from xgboost import XGBRegressor
from omegaconf import DictConfig

from concept_kernels.models.gnn import GNNBaseline
from concept_kernels.models.mlp import RealMLP, MLPSmooth, MLPGSP
from concept_kernels.models.xgb_gsp import XGBGSPRegressor, XGBGSPRegressorSeq
from concept_kernels.models.ssl import SSLFTT, SSLGNN


def build_model(conf: DictConfig,
                kg : Union[Data, None]=None,
                metadata: Union[dict, None] = None):
    if conf.name == "mlp_smooth":
        CLS = MLPSmooth
    elif conf.name == "mlp_gsp":
        CLS = MLPGSP
    elif conf.name == "gnn_baseline":
        CLS = GNNBaseline
    elif conf.name == "ssl_ftt":
        CLS = SSLFTT
    elif conf.name == "ssl_gnn":
        CLS = SSLGNN
    elif conf.name == "xgb_regressor":
        CLS = XGBRegressor
    elif conf.name == "realmlp":
        CLS = RealMLP
    else:
        raise ValueError(f"Wrong model name: {conf.name}")

    return CLS(
        **conf.params,
        task_type=metadata['task_type'],
        num_classes=metadata['num_classes'],
        n_num_features=metadata['n_num_features'],
        n_cat_features=metadata['n_cat_features'],
        kg=kg,
        metadata=metadata
    )
