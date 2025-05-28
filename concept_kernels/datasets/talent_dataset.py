import copy
import json
import logging
import os
import pickle
from pathlib import Path
import typing as ty

import numpy as np
import torch
from torch_geometric.data import Data

from concept_kernels.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'

ArrayDict = ty.Dict[str, np.ndarray]


def load_json(path):
    return json.loads(Path(path).read_text())

def dataname_to_numpy(dataset_name, dataset_path):

    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    dir_ = Path(os.path.join(dataset_path, dataset_name))
    # dir_ = Path(os.path.join(DATA_PATH, dataset_path, dataset_name))

    def load(item) -> ArrayDict:
        return {
            x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle = True))
            for x in ['train', 'val', 'test']
        }

    return (
        load('N') if dir_.joinpath('N_train.npy').exists() else None,
        load('C') if dir_.joinpath('C_train.npy').exists() else None,
        load('y'),
        load_json(dir_ / 'info.json'),
    )


class TalentDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        use_onehot,
        col_embed_model,
        split,
        seed,
    ):
        if use_onehot:
            new_dataset_name = f"{dataset_name}/onehot"
            if os.path.exists(os.path.join(dataset_dir, new_dataset_name)):
                dataset_name = new_dataset_name
            else:
                logger.warn("One-hot converted dataset does not exists. Fall back to the original")
        N, C, y, info = dataname_to_numpy(dataset_name, dataset_dir)
        if info['task_type'] == 'regression':
            info['num_classes'] = -1
        if 'n_classes' in info:
            info['num_classes'] = info['n_classes']

        metadata = copy.deepcopy(info)
        kg = None
        col_embed_fname = 'col_embed_mpnet.pt'
        if col_embed_model == 'sts':
            col_embed_fname = 'col_embed_sts.pt'
        elif col_embed_model == 'qwen' or col_embed_model == 'qwen2':
            col_embed_fname = 'col_embed_qwen.pt'
        elif col_embed_model == 'unif':
            col_embed_fname = 'col_embed_unif.pt'
        embeds_path = os.path.join(dataset_dir, dataset_name, col_embed_fname)
        if os.path.exists(embeds_path):
            col_embeds = torch.load(embeds_path, weights_only=False)
            if isinstance(col_embeds, dict):
                if col_embed_model == 'qwen':
                    col_embeds = torch.nn.functional.normalize(col_embeds['query'] + col_embeds['doc'], p=2, dim=1)
                    col_sim_mat = col_embeds @ col_embeds.T
                elif col_embed_model == 'qwen2':
                    col_sim_mat = col_embeds['doc'] @ col_embeds['query'].T
                    # col_sim_mat = (col_sim_mat + col_sim_mat) / 2
                    # d = col_sim_mat.sum(dim=1) ** -0.5
                    # col_sim_mat = d.unsqueeze(0) * d.unsqueeze(1) * col_sim_mat
                    col_embeds = torch.nn.functional.normalize(col_embeds['query'] + col_embeds['doc'], p=2, dim=1)
            else:
                col_sim_mat = col_embeds @ col_embeds.T
            num_cols = col_embeds.shape[0]
            row_index = torch.arange(num_cols).unsqueeze(0).tile((num_cols, 1))
            col_index = row_index.T.clone()
            kg = Data(
                x=col_embeds,
                edge_index=torch.stack([row_index.flatten(), col_index.flatten()]),
                edge_attr=col_sim_mat.flatten()
            )
            metadata['X_mapping'] = list(range(num_cols))
            metadata['y_mapping'] = -1
            metadata['col_sim_mat'] = col_sim_mat

        super().__init__(
            X_num=N[split] if N is not None else None,
            X_cat=C[split] if C is not None else None,
            y=y[split], kg=kg, metadata=metadata, split=split
        )
