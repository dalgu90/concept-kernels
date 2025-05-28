import argparse
from typing import Union

from omegaconf import OmegaConf, DictConfig, ListConfig


def select(cond, x, y):
    """ A simple OmegaConf resolver that performs select """
    if isinstance(cond, str):
        cond = eval(cond.title()) if cond else False
    return x if cond else y

def append_label(label):
    return f"_{label}" if label else ""

def add_custom_resolvers():
    OmegaConf.register_new_resolver("select", select)
    OmegaConf.register_new_resolver("append_label", append_label)

plato_dataset_map = {
    "BreastCarcinoma": "BC",
    "Chondrosarcoma": "CH",
    "Melanoma": "ME",
    "NonSmallCellLungCarcinoma": "NSCLC",
    "SmallCellLungCarcinoma": "SCLC",
    "BRCA": "BRCA",
    "CM": "CM",
    "CRC": "CRC",
    "NSCLC": "MNSCLC",
    "PDAC": "PDAC",
}

def format_output_dir(conf: DictConfig):
    # Format output_dir here to fill placeholders hard to fill by OmegaConf
    if conf.dataset.name == 'plato_dataset':
        subname = conf.dataset.data_common.dataset_file.split('_')[2]
        dataset_name = f"plato_{plato_dataset_map[subname]}"
    elif conf.dataset.name == 'talent_dataset':
        dataset_name = f"talent_{conf.dataset.data_common.dataset_name}"
    else:
        dataset_name = conf.dataset.name

    model_name = conf.model.name
    if 'label' in conf.model:
        model_name += '_' + conf.model.label
    if conf.model.name.startswith('ssl') and conf.model.params.ssl_loss_type == 'InfoNCE':
        model_name += '_infonce'
    if hasattr(conf, 'label') and conf.label:
        model_name += '_' + conf.label

    kwargs = {
        'dataset_name': dataset_name,
        'model_name': model_name,
    }
    conf.output_dir = conf.output_dir.format(**kwargs)
    if hasattr(conf, 'pretrain_dir'):
        conf.pretrain_dir = conf.pretrain_dir.format(**kwargs)
        if conf.phase == 'pretrain':
            conf.output_dir = conf.pretrain_dir

def change_output_dir(conf: Union[DictConfig, ListConfig, dict, list],
                      new_output_dir: str, new_pretrain_dir: str = None):
    if isinstance(conf, ListConfig) or isinstance(conf, list):
        for v in conf:
            change_output_dir(v, new_output_dir)
    elif isinstance(conf, DictConfig) or isinstance(conf, dict):
        for k, v in conf.items():
            if k == 'pretrain_saver': continue
            if isinstance(v, dict) or OmegaConf.is_config(v):
                change_output_dir(v, new_output_dir)
            elif k in ['output_dir', 'log_dir', 'checkpoint_dir']:
                conf[k] = new_output_dir
