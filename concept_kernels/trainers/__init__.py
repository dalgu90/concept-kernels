from concept_kernels.trainers.xgb_trainer import XGBTrainer
from concept_kernels.trainers.base_trainer import BaseTrainer
from concept_kernels.trainers.ssl_trainer import SSLTrainer


def build_trainer(conf):
    if conf.name == "xgb_trainer":
        CLS = XGBTrainer
    elif conf.name == "base_trainer":
        CLS = BaseTrainer
    elif conf.name == "ssl_trainer":
        CLS = SSLTrainer
    else:
        raise ValueError(f"Wrong trainer name: {conf.name}")

    return CLS(conf.params)
