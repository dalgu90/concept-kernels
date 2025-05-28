import logging
import math
import typing as ty
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data

from concept_kernels.datasets.base_dataset import BaseDataset
from concept_kernels.models.base import BaseDeepModel
from concept_kernels.models.ftt import Transformer

logger = logging.getLogger(__name__)


def D(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}

def info_nce(z1, z2, temp=0.1):
    scores = torch.mm(z1, z2.T) / temp
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=z1.device)
    total_loss = (torch.nn.functional.cross_entropy(scores, labels) + \
                  torch.nn.functional.cross_entropy(scores.T, labels)) / 2.0
    return {'total': total_loss}

class SSLBase(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        kg: Data,
        metadata: dict,
        ssl_loss_type: str = 'SCL',
        tau: float = 0.1,
        num_swap_reg = 'fixcorr',
        **kwargs
    ):
        super().__init__(
            task_type=task_type,
            num_classes=num_classes,
            n_num_features=n_num_features,
            n_cat_features=n_cat_features,
            kg=kg,
            metadata=metadata,
            **kwargs
        )
        self.ssl_loss_type = ssl_loss_type
        self.tau = tau
        self.phase = 'pretrain'
        self.num_swap_a = None
        self.num_swap_b = None
        self.num_swap_reg = num_swap_reg

    # def get_param_groups(self):
        # if self.phase == 'finetune':
            # param_groups = [
                # {'params': self.backbone.parameters(), 'lr_mult': 0.1},
                # {'params': self.predictor.parameters(), 'lr_mult': 1.0},
            # ]
        # else:
            # param_groups = [{'params': self.parameters(), 'lr_mult': 1.0}]
        # return param_groups

    def data_preproc(self, dataset: BaseDataset):
        super().data_preproc(dataset)

        if dataset.X_num is not None and dataset.split == 'train':
            C = dataset.X_num.shape[1]
            if self.num_swap_reg != 'no':
                delattr(self, 'num_swap_a')
                delattr(self, 'num_swap_b')

            if self.num_swap_reg == 'no':
                pass
            elif self.num_swap_reg == 'fixcorr':
                num_swap_a = torch.tensor(np.cov(dataset.X_num.T).reshape((C, C)))
                num_swap_b = torch.zeros((C, C), dtype=torch.float)
                self.register_buffer('num_swap_a', num_swap_a)
                self.register_buffer('num_swap_b', num_swap_b)
            elif self.num_swap_reg == 'corr':
                self.num_swap_a = nn.Parameter(torch.tensor(np.cov(dataset.X_num.T).reshape((C, C))))
                self.num_swap_b = nn.Parameter(torch.zeros((C, C), dtype=torch.float))
            elif self.num_swap_reg == 'fixsign':
                num_swap_a = torch.sign(torch.tensor(np.cov(dataset.X_num.T).reshape((C, C))))
                num_swap_b = torch.zeros((C, C), dtype=torch.float)
                self.register_buffer('num_swap_a', num_swap_a)
                self.register_buffer('num_swap_b', num_swap_b)

    def dumps_encoders(self):
        dumps = super().dumps_encoders()
        dumps['num_swap_a'] = pickle.dumps(self.num_swap_a)
        dumps['num_swap_b'] = pickle.dumps(self.num_swap_b)
        return dumps

    def loads_encoders(self, dumps):
        super().loads_encoders(dumps)
        self.num_swap_a = pickle.loads(dumps['num_swap_a'])
        self.num_swap_b = pickle.loads(dumps['num_swap_b'])

    def swap_regress(self, X_num, perm_num):
        assert X_num is not None and perm_num is not None
        if self.num_swap_reg == 'no':
            return
        B, C = X_num.shape
        # row = torch.arange(B).unsqueeze(1).repeat(1, C).to(X_num.device)
        row = torch.arange(C).unsqueeze(0).repeat(B, 1).to(X_num.device)
        a = torch.where(row == perm_num, 1.0, self.num_swap_a[row, perm_num])
        b = torch.where(row == perm_num, 0.0, self.num_swap_b[row, perm_num])
        X_num *= a
        X_num += b

    def compute_ssl_loss(self, z1, z2):
        loss = {}
        if self.ssl_loss_type == 'SCL':
            L, l = D(z1, z2, mu=1.0)
            loss['total'] = L
            loss.update(l)
        elif self.ssl_loss_type == 'InfoNCE':
            loss = info_nce(z1, z2, temp=self.tau)
        else:
            raise ValueError
        return loss

    def forward(self, batch, compute_loss=False, return_output=False):
        if 'X_num' in batch or 'X_cat' in batch:
            # Supervised learning
            z = self.backbone(batch['X_num'], batch['X_cat'])
            x = self.predictor(z)
            batch_output = x

            if compute_loss:
                loss = self.compute_loss(batch_output, batch['label'])
                if return_output:
                    return loss, batch_output, z
                else:
                    return loss
            else:
                return batch_output, z
        else:
            # Contrastive learning
            assert 'X_num1' in batch or 'X_num2' in batch
            if batch.get('perm_num1', None) is not None:
                self.swap_regress(batch['X_num1'], batch['perm_num1'])
                self.swap_regress(batch['X_num2'], batch['perm_num2'])
            z1 = self.backbone(batch['X_num1'], batch['X_cat1'])
            z2 = self.backbone(batch['X_num2'], batch['X_cat2'])
            batch_output = torch.cat([z1, z2], dim=1)

            if compute_loss:
                loss = self.compute_ssl_loss(z1, z2)
                if return_output:
                    return loss, batch_output, None
                else:
                    return loss
            else:
                return batch_output, None


class SSLFTT(SSLBase):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        # tokenizer
        # d_numerical: int,
        # categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        mlp_num_layers: int,
        mlp_hidden_dim: int,
        dropout: float,
        kg: Data,
        metadata: dict,
        **kwargs
    ):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        super().__init__(
            task_type=task_type,
            num_classes=num_classes,
            n_num_features=n_num_features,
            n_cat_features=n_cat_features,
            kg=kg,
            metadata=metadata,
            **kwargs
        )

        self.token_bias = token_bias
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_ffn_factor = d_ffn_factor
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.activation = activation
        self.prenormalization = prenormalization
        self.initialization = initialization
        self.kv_compression = kv_compression
        self.kv_compression_sharing = kv_compression_sharing
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1
        categories = None
        if 'categories' in metadata and metadata['categories']:
            categories = metadata['categories']

        self.backbone = Transformer(
            d_numerical=n_num_features,
            categories=categories,
            token_bias=token_bias,
            # transformer
            n_layers=n_layers,
            d_token=d_token,
            n_heads=n_heads,
            d_ffn_factor=d_ffn_factor,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            activation=activation,
            prenormalization=prenormalization,
            initialization=initialization,
            # linformer
            kv_compression=kv_compression,
            kv_compression_sharing=kv_compression_sharing,
        )

        layers = []
        for i in range(self.mlp_num_layers):
            in_dim=self.d_token if i == 0 else self.mlp_hidden_dim
            out_dim=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else pred_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.mlp_num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.predictor = nn.Sequential(*layers)


class GNNBackbone(nn.Module):
    def __init__(
        self,
        input_embed_dim: int,
        conv_layer_type: str,
        conv_num_layers: int,
        conv_hidden_dim: int,
        gat_num_heads: int,
        conv_pool_method: str,
        edge_active_ratio: float,
        kg: Data,
        metadata: dict,
    ):
        super().__init__()

        self.input_embed_dim = input_embed_dim
        self.conv_layer_type = conv_layer_type
        self.conv_num_layers = conv_num_layers
        self.conv_hidden_dim = conv_hidden_dim
        self.gat_num_heads = gat_num_heads
        self.conv_pool_method = conv_pool_method
        self.edge_active_ratio = edge_active_ratio
        self.kg = kg
        self.metadata = metadata

        # Select edge_active_ratio of the pos/neg edges with large magnitude
        V, edge_index, edge_weight = kg.x, kg.edge_index, kg.edge_attr
        w_pos = sorted(edge_weight[edge_weight > 0.0].tolist(), reverse=True)
        w_neg = sorted(edge_weight[edge_weight < 0.0].tolist())
        assert self.edge_active_ratio > 0.0
        x_pos = w_pos[math.ceil(len(w_pos)*self.edge_active_ratio)-1]
        x_neg = w_neg[math.ceil(len(w_neg)*self.edge_active_ratio)-1] if w_neg else -1e8
        edge_mask = (edge_weight >= x_pos) | (edge_weight <= x_neg)
        edge_index = edge_index[:, edge_mask]
        self.register_buffer("V", V)
        self.register_buffer("edge_index", edge_index)

        # Input embedding
        # self.input_embed = nn.Parameter(torch.zeros(V.shape[0], self.input_embed_dim))
        self.embed_fc = nn.Linear(V.shape[1], self.input_embed_dim)

        for i in range(self.conv_num_layers):
            if self.conv_layer_type == 'GCNConv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim
                # in_dim = self.V.shape[1] if i == 0 else self.conv_hidden_dim
                out_dim = self.conv_hidden_dim
                conv = pyg.nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
            elif self.conv_layer_type == 'GATConv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim * self.gat_num_heads
                # in_dim = self.V.shape[1] if i == 0 else self.conv_hidden_dim
                out_dim = self.conv_hidden_dim
                conv = pyg.nn.GATConv(in_channels=in_dim, out_channels=out_dim, heads=self.gat_num_heads)
            setattr(self, f"conv{i+1}", conv)

    def forward(self, X_num, X_cat):
        x = [X_num, X_cat]
        x = torch.cat([t for t in x if t is not None], dim=1)

        x = x.unsqueeze(2) * self.embed_fc(self.V).unsqueeze(0)
        # x = x.unsqueeze(2) * self.V.unsqueeze(0)

        B, D = x.shape[:2]
        E = self.edge_index.shape[1]
        x = x.view(B * D, -1)
        edge_index = self.edge_index.repeat(1, B)
        edge_index += torch.arange(B, device=edge_index.device).repeat_interleave(E) * D
        for j in range(self.conv_num_layers):
            conv = getattr(self, f"conv{j+1}")
            x = conv(x, edge_index=self.edge_index)
            x = F.relu(x)
        x = x.view(B, D, -1)
        if self.conv_pool_method == 'max':
            x, _ = x.max(dim=1)
        elif self.conv_pool_method == 'mean':
            x = x.mean(dim=1)

        return x

    def reset_parameters(self):
        if hasattr(self, 'input_feature'):
            nn.init.normal_(self.input_embed, std=1.0/self.input_embed_dim**0.5)
        for i in range(self.conv_num_layers):
            getattr(self, f"conv{i+1}").reset_parameters()


class SSLGNN(SSLBase):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        input_embed_dim: int,
        conv_layer_type: str,
        conv_num_layers: int,
        conv_hidden_dim: int,
        gat_num_heads: int,
        conv_pool_method: str,
        edge_active_ratio: float,
        mlp_num_layers: int,
        mlp_hidden_dim: int,
        dropout: float,
        kg: Data,
        metadata: dict,
        **kwargs
    ):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        super().__init__(
            task_type=task_type,
            num_classes=num_classes,
            n_num_features=n_num_features,
            n_cat_features=n_cat_features,
            kg=kg,
            metadata=metadata,
            **kwargs
        )

        self.input_embed_dim = input_embed_dim
        self.conv_layer_type = conv_layer_type
        self.conv_num_layers = conv_num_layers
        self.conv_hidden_dim = conv_hidden_dim
        self.gat_num_heads = gat_num_heads
        self.conv_pool_method = conv_pool_method
        self.edge_active_ratio = edge_active_ratio
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        self.backbone = GNNBackbone(
            input_embed_dim,
            conv_layer_type,
            conv_num_layers,
            conv_hidden_dim,
            gat_num_heads,
            conv_pool_method,
            edge_active_ratio,
            kg,
            metadata,
        )
        self.backbone.reset_parameters()

        layers = []
        for i in range(self.mlp_num_layers):
            if self.conv_layer_type == 'GCNConv':
                in_dim=self.conv_hidden_dim if i == 0 else self.mlp_hidden_dim
            elif self.conv_layer_type == 'GATConv':
                in_dim=self.conv_hidden_dim * self.gat_num_heads if i == 0 else self.mlp_hidden_dim
            out_dim=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else pred_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.mlp_num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.predictor = nn.Sequential(*layers)
