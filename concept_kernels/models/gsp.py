import logging
import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
import sklearn.cluster

from concept_kernels.models.base import BaseDeepModel
from concept_kernels.models.mlp import ScalingLayer, PLREmbeddingLayer, NTPLinear
from concept_kernels.utils.graph_utils import (
    make_symmetric, get_normalized_laplacian, get_topk_eigen,
    get_global_eig_vecs, get_local_eig_vecs
)

logger = logging.getLogger(__name__)


class GSPBaseline(nn.Module):
    def __init__(
        self,
        gsp_num_comps: int,
        gsp_num_layers: int,
        mlp_num_layers: int,
        mlp_hidden_dim: int,
        input_method: str,
        input_hidden_dim: int,
        kg_subgraph: bool,
        edge_weight_method: str,
        edge_threshold: float,
        l1_weight: float,
        l2_weight: float,
        kg: Data,
        metadata: dict,
        **kwargs
    ):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        super(GSPBaseline, self).__init__()
        # kg provides:
        #   - kg.x: the node embeddings of KG
        #   - kg.edge_index, kg.edge_attr: edges
        #   - kg.relation_embedding: relation_embedding (of ComplEx)
        # metadata provides:
        #   - metadata['X_mapping']: node indices of columns in the KG

        # TODO: build graph for GNN to work on
        #  option 1. edge_ij = cos e_i e_j * edge_ij from kg.edge_index
        #  option 1-2. edge_ij = (ComplEx score of ij) * edge_ij from kg.edge_index
        #  option 2. edge_ij = I(cos e_i e_j > thres)

        # Define GCN and output layer (single number)

        self.gsp_num_comps = gsp_num_comps
        self.gsp_num_layers = gsp_num_layers
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.input_method = input_method
        self.input_hidden_dim = input_hidden_dim
        self.kg_subgraph = kg_subgraph
        self.edge_weight_method = edge_weight_method
        self.edge_threshold = edge_threshold
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.kg = kg
        self.metadata = metadata

        if self.input_method == 'kge':
            self.in_dim = kg.num_node_features
        elif self.input_method == 'feature':
            self.in_dim = self.input_hidden_dim
            self.input_feature = nn.Parameter(
                torch.zeros(len(metadata['X_mapping']), self.input_hidden_dim)
            )
        elif self.input_method == 'concat':
            self.in_dim = kg.num_node_features + self.input_hidden_dim
            self.input_feature = nn.Parameter(
                torch.zeros(len(metadata['X_mapping']), self.input_hidden_dim)
            )

        # TODO: Option to use all the nodes of KG
        assert self.kg_subgraph == True
        V = kg.x[metadata['X_mapping']]
        edge_index, edge_attr = pyg.utils.subgraph(
            metadata['X_mapping'], kg.edge_index, kg.edge_attr, relabel_nodes=True)
        self.register_buffer("V", V)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        if self.edge_weight_method == 'uniform':
            self.edge_weight = None
        elif self.edge_weight_method == 'cosine':
            assert self.kg_subgraph == True
            from_V, to_V = V[edge_index[0]], V[edge_index[1]]
            edge_weight = (from_V*to_V).sum(dim=1) / from_V.norm(dim=1) / to_V.norm(dim=1)
            self.register_buffer("edge_weight", edge_weight)
        elif self.edge_weight_method == 'score':
            from_V, to_V = V[edge_index[0]], V[edge_index[1]]
            edge_weight = (from_V*kg.relation_embedding[edge_attr]*to_V).sum(dim=1)
            edge_weight = torch.log(1.0 + torch.clip(edge_weight, min=0.0))
            self.register_buffer("edge_weight", edge_weight)

        # Compute the eigen vectors
        edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
        num_nodes = V.shape[0]
        laplacian = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
        eig_vals, eig_vecs = get_topk_eigen(laplacian, self.gsp_num_comps)
        self.register_buffer("eig_vecs", eig_vecs)

        for i in range(self.gsp_num_layers):
            spec_filter = nn.Parameter(torch.ones(self.gsp_num_comps, self.in_dim))
            setattr(self, f"gsp_filter{i+1}", spec_filter)
            mix_fc = nn.Linear(in_features=self.in_dim, out_features=self.in_dim)
            setattr(self, f"gsp_fc{i+1}", mix_fc)
        # self.pool = pyg.nn.global_mean_pool
        for i in range(self.mlp_num_layers):
            fc = nn.Linear(
                in_features=self.in_dim if i == 0 else self.mlp_hidden_dim,
                out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            )
            setattr(self, f"fc{i+1}", fc)

        self.reset_parameters()

    def forward(self, batch, compute_loss=False, return_output=False):
        x = batch['input']  # [batch_size, num_cols]
        batch_size = x.shape[0]
        if self.input_method == 'kge':
            x = self.V.unsqueeze(0) * x.unsqueeze(2)  # [batch_size, num_cols, in_dim]
        elif self.input_method == 'feature':
            x = self.input_feature.unsqueeze(0) * x.unsqueeze(2)
        elif self.input_method == 'concat':
            x = torch.cat([
                self.input_feature.unsqueeze(0) * x.unsqueeze(2),
                self.V.unsqueeze(0).repeat(batch_size, 1, 1),
            ], dim=2)


        x = torch.einsum('bnd,nk->bkd', x, self.eig_vecs)
        for i in range(self.gsp_num_layers):
            spec_filter = getattr(self, f"gsp_filter{i+1}")
            mix_fc = getattr(self, f"gsp_fc{i+1}")
            x = F.relu(x * spec_filter.unsqueeze(0))
            x = F.relu(mix_fc(x))
        x = torch.einsum('bkd,nk->bnd', x, self.eig_vecs)
        x = F.relu(x)
        x = x.mean(dim=1)
        # x = x.max(dim=1)[0]
        for j in range(self.mlp_num_layers):
            fc = getattr(self, f"fc{j+1}")
            x = fc(x)
            if j != self.mlp_num_layers - 1:
                x = F.relu(x)

        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def compute_loss(self, pred, label):
        loss = 0.0
        ret = {}

        mse_loss = F.mse_loss(pred.flatten(), label)
        loss += mse_loss
        ret['mse'] = mse_loss

        if self.l1_weight:
            l1_loss = 0.0
            for parameter in self.parameters():
                l1_loss += self.l1_weight * parameter.abs().sum()
            loss += l1_loss
            ret['l1'] = l1_loss

        if self.l2_weight:
            l2_loss = 0.0
            for parameter in self.parameters():
                l2_loss += self.l2_weight * parameter.square().sum()
            loss += l2_loss
            ret['l2'] = l2_loss

        ret['total'] = loss
        return ret

    def reset_parameters(self):
        if self.input_method == 'concat':
            nn.init.normal_(self.input_feature, std=1.0/self.input_hidden_dim**0.5)
        for i in range(self.gsp_num_layers):
            getattr(self, f"gsp_fc{i+1}").reset_parameters()
            nn.init.ones_(getattr(self, f"gsp_filter{i+1}"))
        for i in range(self.mlp_num_layers):
            getattr(self, f"fc{i+1}").reset_parameters()


class GSPBaselineSimple(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        gsp_matrix_type: str,
        gsp_num_comps: int,
        gsp_center_embed: bool,
        num_layers: int,
        hidden_dim: int,
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

        self.gsp_matrix_type = gsp_matrix_type
        self.gsp_num_comps = gsp_num_comps
        self.gsp_center_embed = gsp_center_embed
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        feature_dim = self.n_num_features + self.n_cat_features
        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        assert self.kg is not None and 'X_mapping' in self.metadata
        assert feature_dim == len(self.metadata['X_mapping'])

        # num_nodes = kg.num_nodes
        # edge_index = kg.edge_index
        # edge_weight = kg.edge_attr
        # X_mapping = self.metadata['X_mapping']
        # edge_index, edge_weight = pyg.utils.subgraph(
            # X_mapping, edge_index, edge_weight, relabel_nodes=True
        # )
        # edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
        # if self.gsp_matrix_type == 'adj':
            # A = torch.zeros([num_nodes, num_nodes])
            # A[edge_index[0], edge_index[1]] = edge_weight
        # elif self.gsp_matrix_type == 'lap':
            # A = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
        # eig_vals, eig_vecs = get_topk_eigen(
            # A,
            # min(self.gsp_num_comps, feature_dim),
            # largest=(self.gsp_matrix_type=='adj')
        # )

        X = kg.x
        num_comps = min(self.gsp_num_comps, feature_dim)
        if self.gsp_center_embed:
            X = X - X.mean(dim=0, keepdims=True)

        eig_vecs = self.compute_eigen_vectors(X, num_comps, self.gsp_matrix_type)
        self.register_buffer("eig_vecs", eig_vecs)

        layers = []
        for i in range(self.num_layers):
            in_dim = num_comps if i == 0 else self.hidden_dim
            out_dim = pred_dim if i == self.num_layers-1 else self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(*layers)

    def compute_eigen_vectors(self, col_embed, topk, gsp_matrix_type='adj'):
        col_embed = F.normalize(col_embed, p=2, dim=1)
        W = col_embed @ col_embed.T
        if gsp_matrix_type == 'adj':
            eig_vals, eig_vecs = get_topk_eigen(W, topk, True)
        elif gsp_matrix_type == 'lap':
            W = torch.clamp(W, min=0.0)
            d = W.sum(dim=0)
            d_sqrt = torch.where(d > 1e-8, d ** -0.5, torch.zeros_like(d))
            L = torch.eye(W.shape[0]) - d_sqrt.unsqueeze(1) * W * d_sqrt.unsqueeze(0)
            # L = torch.diag(W.sum(dim=0)) - W
            eig_vals, eig_vecs = get_topk_eigen(L, topk, False)
        return eig_vecs

    def forward(self, batch, compute_loss=False, return_output=False):
        x = [batch['X_num'], batch['X_cat']]
        x = torch.cat([t for t in x if t is not None], dim=1)

        x = torch.einsum('bn,nk->bk', x, self.eig_vecs)
        x = self.mlp(x)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output


class GSPBaselineCluster(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        gsp_matrix_type: str,
        gsp_num_comps: int,
        gsp_cluster_method: str,
        use_raw_kernel: bool,
        num_layers: int,
        hidden_dim: int,
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

        self.gsp_matrix_type = gsp_matrix_type
        self.gsp_num_comps = gsp_num_comps
        self.gsp_cluster_method = gsp_cluster_method
        self.use_raw_kernel = use_raw_kernel
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        feature_dim = self.n_num_features + self.n_cat_features
        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        assert self.kg is not None and 'X_mapping' in self.metadata
        assert feature_dim == len(self.metadata['X_mapping'])

        X = kg.x
        num_comps = min(self.gsp_num_comps, feature_dim)

        eig_vecs = []
        if self.use_raw_kernel:
            eig_vecs.append(self.compute_eigen_vectors(X, num_comps, self.gsp_matrix_type, None))
        eig_vecs.append(self.compute_eigen_vectors(X, num_comps, self.gsp_matrix_type, self.gsp_cluster_method))
        eig_vecs = torch.concat(eig_vecs, dim=1)
        self.register_buffer("eig_vecs", eig_vecs)

        layers = []
        for i in range(self.num_layers):
            in_dim = eig_vecs.shape[1] if i == 0 else self.hidden_dim
            out_dim = pred_dim if i == self.num_layers-1 else self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(*layers)

    def compute_eigen_vectors(self, col_embed, topk, gsp_matrix_type='adj', gsp_cluster_method=None):
        col_embed = F.normalize(col_embed, p=2, dim=1)
        W = self.compute_cluster_kernel(col_embed, gsp_cluster_method)
        if gsp_matrix_type == 'adj':
            eig_vals, eig_vecs = get_topk_eigen(W, topk, True)
        elif gsp_matrix_type == 'lap':
            # W = torch.clamp(W, min=0.0)
            # d = W.sum(dim=0)
            # d_sqrt = torch.where(d > 1e-8, d ** -0.5, torch.zeros_like(d))
            # L = torch.eye(W.shape[0]) - d_sqrt.unsqueeze(1) * W * d_sqrt.unsqueeze(0)
            L = torch.diag(W.sum(dim=0)) - W
            eig_vals, eig_vecs = get_topk_eigen(L, topk, False)
        return eig_vecs

    def compute_cluster_kernel(self, col_embed, gsp_cluster_method=None):
        W = col_embed @ col_embed.T
        n = col_embed.shape[0]
        if gsp_cluster_method is not None:
            if gsp_cluster_method == 'DBSCAN':
                estimator = sklearn.cluster.DBSCAN(metric='cosine', min_samples=2, eps=0.3)
            elif gsp_cluster_method == 'HDBSCAN':
                estimator = sklearn.cluster.HDBSCAN(metric='cosine', min_cluster_size=2)
            elif gsp_cluster_method == 'KMeans':
                estimator = sklearn.cluster.KMeans(n_clusters=math.ceil(n**0.5))
            elif gsp_cluster_method == 'SpectralClustering':
                estimator = sklearn.cluster.SpectralClustering(n_clusters=math.ceil(n**0.5))
            labels = estimator.fit_predict(col_embed)
            max_label = max(labels)
            for i in range(max_label+1):
                idxs = np.where(labels == i)[0]
                sel_embed = col_embed[idxs]
                sel_embed -= sel_embed.mean(dim=0, keepdims=True)
                sel_embed = F.normalize(sel_embed, p=2, dim=1)
                W[np.ix_(idxs, idxs)] = sel_embed @ sel_embed.T
        return W

    def forward(self, batch, compute_loss=False, return_output=False):
        x = [batch['X_num'], batch['X_cat']]
        x = torch.cat([t for t in x if t is not None], dim=1)

        x = torch.einsum('bn,nk->bk', x, self.eig_vecs)
        x = self.mlp(x)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output


class GSPBaselineMulti(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        gsp_matrix_type: str,
        gsp_num_comps: int,
        gsp_num_projs: int,
        gsp_center_embed: bool,
        num_layers: int,
        hidden_dim: int,
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

        self.gsp_matrix_type = gsp_matrix_type
        self.gsp_num_projs = gsp_num_projs
        self.gsp_num_comps = gsp_num_comps
        self.gsp_center_embed = gsp_center_embed
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        feature_dim = self.n_num_features + self.n_cat_features
        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        assert self.kg is not None and 'X_mapping' in self.metadata
        assert feature_dim == len(self.metadata['X_mapping'])

        X = kg.x
        num_comps = min(self.gsp_num_comps, feature_dim)
        embed_dim = X.shape[1]
        if self.gsp_center_embed:
            X = X - X.mean(dim=0, keepdims=True)
        self.register_buffer("col_embeds", X)

        eig_vecs = self.compute_eigen_vectors(X, num_comps, self.gsp_matrix_type)
        self.register_buffer("eig_vecs", eig_vecs)
        for i in range(self.gsp_num_projs):
            proj = nn.Parameter(torch.zeros(embed_dim))
            setattr(self, f"proj_{i+1}", proj)

        layers = []
        for i in range(self.num_layers):
            in_dim = (self.gsp_num_projs+1)*num_comps if i == 0 else self.hidden_dim
            out_dim = pred_dim if i == self.num_layers-1 else self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.gsp_num_projs):
            nn.init.normal_(getattr(self, f"proj_{i+1}"))

    def compute_eigen_vectors(self, col_embed, topk, gsp_matrix_type='adj'):
        col_embed = F.normalize(col_embed, p=2, dim=1)
        W = col_embed @ col_embed.T
        if gsp_matrix_type == 'adj':
            eig_vals, eig_vecs = get_topk_eigen(W, topk, True)
        elif gsp_matrix_type == 'lap':
            # W = torch.clamp(W, min=0.0)
            # d = W.sum(dim=0)
            # d_sqrt = torch.where(d > 1e-8, d ** -0.5, torch.zeros_like(d))
            # L = torch.eye(W.shape[0], device=W.device) - d_sqrt.unsqueeze(1) * W * d_sqrt.unsqueeze(0)
            L = torch.diag(W.sum(dim=0)) - W
            eig_vals, eig_vecs = get_topk_eigen(L, topk, False)
        return eig_vecs

    def forward(self, batch, compute_loss=False, return_output=False):
        x = [batch['X_num'], batch['X_cat']]
        x = torch.cat([t for t in x if t is not None], dim=1)
        num_comps = min(self.gsp_num_comps, x.shape[1])

        # self.proj_embeds = []
        # self.eig_vecss = []
        self.comps = [torch.einsum('bn,nk->bk', x, self.eig_vecs)]
        for i in range(self.gsp_num_projs):
            proj_embed = self.col_embeds * getattr(self, f"proj_{i+1}").unsqueeze(0)
            eig_vecs = self.compute_eigen_vectors(proj_embed, num_comps, self.gsp_matrix_type)
            comp = torch.einsum('bn,nk->bk', x, eig_vecs)
            # comp.retain_grad()
            # eig_vecs.retain_grad()
            # proj_embed.retain_grad()
            # self.proj_embeds.append(proj_embed)
            self.comps.append(comp)
            # self.eig_vecss.append(eig_vecs)
        x = torch.concat(self.comps, dim=1)

        x = self.mlp(x)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output


class GSPBaselineMulti2(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        gsp_matrix_type: str,
        gsp_num_comps: int,
        gsp_num_projs: int,
        gsp_proj_dim: int,
        gsp_center_embed: bool,
        num_layers: int,
        hidden_dim: int,
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

        self.gsp_matrix_type = gsp_matrix_type
        self.gsp_num_projs = gsp_num_projs
        self.gsp_proj_dim = gsp_proj_dim
        self.gsp_num_comps = gsp_num_comps
        self.gsp_center_embed = gsp_center_embed
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        feature_dim = self.n_num_features + self.n_cat_features
        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        assert self.kg is not None and 'X_mapping' in self.metadata
        assert feature_dim == len(self.metadata['X_mapping'])

        X = kg.x
        num_comps = min(self.gsp_num_comps, feature_dim)
        embed_dim = X.shape[1]
        if self.gsp_center_embed:
            X = X - X.mean(dim=0, keepdims=True)
        self.register_buffer("col_embeds", X)

        eig_vecs = self.compute_eigen_vectors(X, num_comps, self.gsp_matrix_type)
        self.register_buffer("eig_vecs", eig_vecs)
        for i in range(self.gsp_num_projs):
            setattr(self, f"proj_{i+1}", nn.Linear(embed_dim, self.gsp_proj_dim))

        layers = []
        for i in range(self.num_layers):
            in_dim = (self.gsp_num_projs+1)*num_comps if i == 0 else self.hidden_dim
            out_dim = pred_dim if i == self.num_layers-1 else self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(*layers)

    def compute_eigen_vectors(self, col_embed, topk, gsp_matrix_type='adj'):
        col_embed = F.normalize(col_embed, p=2, dim=1)
        W = col_embed @ col_embed.T
        if gsp_matrix_type == 'adj':
            eig_vals, eig_vecs = get_topk_eigen(W, topk, True)
        elif gsp_matrix_type == 'lap':
            # W = torch.clamp(W, min=0.0)
            # d = W.sum(dim=0)
            # d_sqrt = torch.where(d > 1e-8, d ** -0.5, torch.zeros_like(d))
            # L = torch.eye(W.shape[0], device=W.device) - d_sqrt.unsqueeze(1) * W * d_sqrt.unsqueeze(0)
            L = torch.diag(W.sum(dim=0)) - W
            eig_vals, eig_vecs = get_topk_eigen(L, topk, False)
        return eig_vecs

    def forward(self, batch, compute_loss=False, return_output=False):
        x = [batch['X_num'], batch['X_cat']]
        x = torch.cat([t for t in x if t is not None], dim=1)
        num_comps = min(self.gsp_num_comps, x.shape[1])

        # self.proj_embeds = []
        # self.eig_vecss = []
        self.comps = [torch.einsum('bn,nk->bk', x, self.eig_vecs)]
        for i in range(self.gsp_num_projs):
            proj_embed = getattr(self, f"proj_{i+1}")(self.col_embeds)
            eig_vecs = self.compute_eigen_vectors(proj_embed, num_comps, self.gsp_matrix_type)
            comp = torch.einsum('bn,nk->bk', x, eig_vecs)
            # comp.retain_grad()
            # eig_vecs.retain_grad()
            # proj_embed.retain_grad()
            # self.proj_embeds.append(proj_embed)
            self.comps.append(comp)
            # self.eig_vecss.append(eig_vecs)
        x = torch.concat(self.comps, dim=1)

        x = self.mlp(x)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output



# class GSPBaselineSimple(nn.Module):
    # def __init__(
        # self,
        # graph_scope: str,
        # edge_weight_method: str,
        # gsp_matrix_type: str,
        # gsp_num_comps: int,
        # gene_num_layers: int,
        # gene_hidden_dim: int,
        # drug_num_layers: int,
        # drug_hidden_dim: int,
        # fusion_num_layers: int,
        # fusion_hidden_dim: int,
        # dropout: float,
        # l1_weight: float,
        # l2_weight: float,
        # kg: Data,
        # metadata: dict,
        # **kwargs
    # ):
        # cls_name = self.__class__.__name__
        # logger.info(f"Initializing {cls_name}")
        # super(GSPBaselineSimple, self).__init__()
        # # kg provides:
        # #   - kg.x: the node embeddings of KG
        # #   - kg.edge_index, kg.edge_attr: edges
        # #   - kg.relation_embeddings: relation_embeddings (of ComplEx)
        # # metadata provides:
        # #   - metadata['X_mapping']: node indices of columns in the KG

        # self.graph_scope = graph_scope
        # self.edge_weight_method = edge_weight_method
        # self.gsp_matrix_type = gsp_matrix_type
        # self.gsp_num_comps = gsp_num_comps
        # self.gene_num_layers = gene_num_layers
        # self.gene_hidden_dim = gene_hidden_dim
        # self.drug_num_layers = drug_num_layers
        # self.drug_hidden_dim = drug_hidden_dim
        # self.fusion_num_layers = fusion_num_layers
        # self.fusion_hidden_dim = fusion_hidden_dim
        # self.dropout = dropout
        # self.l1_weight = l1_weight
        # self.l2_weight = l2_weight
        # self.kg = kg
        # self.metadata = metadata

        # if self.graph_scope == 'global':
            # eig_vecs = get_global_eig_vecs(
                # self.edge_weight_method,
                # self.gsp_matrix_type,
                # self.gsp_num_comps
            # )
            # eig_vecs = eig_vecs[:self.metadata['num_gene_cols']]
        # elif self.graph_scope == 'local' or self.graph_scope == 'gene':
            # eig_vecs = get_local_eig_vecs(
                # self.kg,
                # self.metadata,
                # self.graph_scope == 'gene',
                # self.edge_weight_method,
                # self.gsp_matrix_type,
                # self.gsp_num_comps
            # )
            # if self.graph_scope == 'local':
                # eig_vecs = eig_vecs[:self.metadata['num_gene_cols']]
        # self.register_buffer("eig_vecs", eig_vecs)

        # gene_in_dim = self.gsp_num_comps
        # gene_out_dim = 1 if self.fusion_num_layers == 0 else (
            # self.gene_hidden_dim if self.gene_num_layers > 0 else gene_in_dim)
        # if self.dropout:
            # self.gene_dropout = nn.Dropout(self.dropout)
        # for i in range(self.gene_num_layers):
            # fc = nn.Linear(
                # in_features=gene_in_dim if i == 0 else self.gene_hidden_dim,
                # out_features=gene_out_dim if i == self.gene_num_layers - 1 else self.gene_hidden_dim
            # )
            # setattr(self, f"gene_fc{i+1}", fc)

        # drug_in_dim = self.metadata['num_drug_cols']
        # drug_out_dim = 1 if self.fusion_num_layers == 0 else (
            # self.drug_hidden_dim if self.drug_num_layers > 0 else drug_in_dim)
        # for i in range(self.drug_num_layers):
            # fc = nn.Linear(
                # in_features=drug_in_dim if i == 0 else self.drug_hidden_dim,
                # out_features=drug_out_dim if i == self.drug_num_layers - 1 else self.drug_hidden_dim
            # )
            # setattr(self, f"drug_fc{i+1}", fc)

        # fusion_in_dim = gene_out_dim + drug_out_dim
        # if self.fusion_num_layers > 0 and self.dropout:
            # self.fusion_dropout = nn.Dropout(self.dropout)
        # for i in range(self.fusion_num_layers):
            # fc = nn.Linear(
                # in_features=fusion_in_dim if i == 0 else self.fusion_hidden_dim,
                # out_features=1 if i == self.fusion_num_layers - 1 else self.fusion_hidden_dim
            # )
            # setattr(self, f"fusion_fc{i+1}", fc)

        # self.reset_parameters()

    # def forward(self, batch, compute_loss=False, return_output=False):
        # x = batch['input']  # [batch_size, num_cols]
        # batch_size = x.shape[0]

        # gene_x = x[:, :self.metadata['num_gene_cols']]
        # gene_x = torch.einsum('bn,nk->bk', gene_x, self.eig_vecs)
        # for i in range(self.gene_num_layers):
            # gene_x = getattr(self, f"gene_fc{i+1}")(gene_x)
            # if i != self.gene_num_layers - 1:
                # gene_x = F.relu(gene_x)

        # drug_x = x[:, -self.metadata['num_drug_cols']:]
        # for i in range(self.drug_num_layers):
            # drug_x = getattr(self, f"drug_fc{i+1}")(drug_x)
            # if i != self.drug_num_layers - 1:
                # drug_x = F.relu(drug_x)

        # if self.fusion_num_layers == 0:
            # x = gene_x + drug_x
        # else:
            # x = torch.cat([gene_x, drug_x], dim=1)
            # x = self.fusion_dropout(x)
            # for i in range(self.fusion_num_layers):
                # x = getattr(self, f"fusion_fc{i+1}")(x)
                # if i != self.fusion_num_layers - 1:
                    # x = F.relu(x)

        # batch_output = x

        # if compute_loss:
            # loss = self.compute_loss(batch_output, batch['label'])
            # if return_output:
                # return loss, batch_output
            # else:
                # return loss
        # else:
            # return batch_output

    # def compute_loss(self, pred, label):
        # loss = 0.0
        # ret = {}

        # mse_loss = F.mse_loss(pred.flatten(), label)
        # loss += mse_loss
        # ret['mse'] = mse_loss

        # if self.l1_weight:
            # l1_loss = 0.0
            # for parameter in self.parameters():
                # l1_loss += self.l1_weight * parameter.abs().sum()
            # loss += l1_loss
            # ret['l1'] = l1_loss

        # if self.l2_weight:
            # l2_loss = 0.0
            # for parameter in self.parameters():
                # l2_loss += self.l2_weight * parameter.square().sum()
            # loss += l2_loss
            # ret['l2'] = l2_loss

        # ret['total'] = loss
        # return ret

    # def reset_parameters(self):
        # for i in range(self.gene_num_layers):
            # getattr(self, f"gene_fc{i+1}").reset_parameters()
        # for i in range(self.drug_num_layers):
            # getattr(self, f"drug_fc{i+1}").reset_parameters()
        # for i in range(self.fusion_num_layers):
            # getattr(self, f"fusion_fc{i+1}").reset_parameters()


# class GSPBaselineSimpleSplit(nn.Module):
    # def __init__(
        # self,
        # gsp_num_comps: int,
        # gsp_num_layers: int,
        # mlp_num_layers: int,
        # mlp_hidden_dim: int,
        # kg_subgraph: bool,
        # edge_weight_method: str,
        # edge_threshold: float,
        # l1_weight: float,
        # l2_weight: float,
        # kg: Data,
        # metadata: dict,
        # **kwargs
    # ):
        # cls_name = self.__class__.__name__
        # logger.info(f"Initializing {cls_name}")
        # super(GSPBaselineSimpleSplit, self).__init__()
        # # kg provides:
        # #   - kg.x: the node embeddings of KG
        # #   - kg.edge_index, kg.edge_attr: edges
        # #   - kg.relation_embeddings: relation_embeddings (of ComplEx)
        # # metadata provides:
        # #   - metadata['X_mapping']: node indices of columns in the KG

        # # TODO: build graph for GNN to work on
        # #  option 1. edge_ij = cos e_i e_j * edge_ij from kg.edge_index
        # #  option 1-2. edge_ij = (ComplEx score of ij) * edge_ij from kg.edge_index
        # #  option 2. edge_ij = I(cos e_i e_j > thres)

        # # Define GCN and output layer (single number)

        # self.gsp_num_comps = gsp_num_comps
        # self.gsp_num_layers = gsp_num_layers
        # self.mlp_num_layers = mlp_num_layers
        # self.mlp_hidden_dim = mlp_hidden_dim
        # self.kg_subgraph = kg_subgraph
        # self.edge_weight_method = edge_weight_method
        # self.edge_threshold = edge_threshold
        # self.l1_weight = l1_weight
        # self.l2_weight = l2_weight
        # self.kg = kg
        # self.metadata = metadata

        # # TODO: Option to use all the nodes of KG
        # assert self.kg_subgraph == True
        # X_mapping = metadata['X_mapping']
        # X_mapping = X_mapping[:-metadata['num_drug_cols']]
        # V = kg.x[X_mapping]
        # edge_index, edge_attr = pyg.utils.subgraph(
            # X_mapping, kg.edge_index, kg.edge_attr, relabel_nodes=True)
        # self.register_buffer("V", V)
        # self.register_buffer("edge_index", edge_index)
        # self.register_buffer("edge_attr", edge_attr)

        # if self.edge_weight_method == 'uniform':
            # self.edge_weight = None
        # elif self.edge_weight_method == 'cosine':
            # assert self.kg_subgraph == True
            # from_V, to_V = V[edge_index[0]], V[edge_index[1]]
            # edge_weight = (from_V*to_V).sum(dim=1) / from_V.norm(dim=1) / to_V.norm(dim=1)
            # self.register_buffer("edge_weight", edge_weight)
        # elif self.edge_weight_method == 'score':
            # from_V, to_V = V[edge_index[0]], V[edge_index[1]]
            # edge_weight = (from_V*kg.relation_embedding[edge_attr]*to_V).sum(dim=1)
            # edge_weight = torch.log(1.0 + torch.clip(edge_weight, min=0.0))
            # self.register_buffer("edge_weight", edge_weight)

        # # Compute the eigen vectors
        # eig_vec_cache_path = f"/tmp/eig_{edge_weight_method}_{metadata['num_gene_cols']}.pt"
        # if os.path.exists(eig_vec_cache_path):
            # eig_vecs = torch.load(eig_vec_cache_path, weights_only=False)
        # else:
            # edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
            # num_nodes = V.shape[0]
            # laplacian = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
            # eig_vals, eig_vecs = get_topk_eigen(laplacian, 1000)
            # torch.save(eig_vecs, eig_vec_cache_path)
        # eig_vecs = eig_vecs[:, :self.gsp_num_comps]
        # self.register_buffer("eig_vecs", eig_vecs)

        # gene_fc_in_dim = self.gsp_num_comps
        # drug_fc_in_dim = metadata['num_drug_cols']

        # for i in range(self.gsp_num_layers):
            # fc = nn.Linear(in_features=self.gsp_num_comps, out_features=self.gsp_num_comps)
            # setattr(self, f"gsp_fc{i+1}", fc)
        # # self.pool = pyg.nn.global_mean_pool
        # for i in range(self.mlp_num_layers):
            # gene_fc = nn.Linear(
                # in_features=gene_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"gene_fc{i+1}", gene_fc)
            # drug_fc = nn.Linear(
                # in_features=drug_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"drug_fc{i+1}", drug_fc)

        # self.reset_parameters()

    # def forward(self, batch, compute_loss=False, return_output=False):
        # x = batch['input']  # [batch_size, num_cols]

        # x = x[:, :-self.metadata['num_drug_cols']]
        # x = torch.einsum('bn,nk->bk', x, self.eig_vecs)
        # for i in range(self.gsp_num_layers):
            # fc = getattr(self, f"gsp_fc{i+1}")
            # x = F.relu(fc(x))
        # h_gene = x
        # h_drug = batch['input'][:, -self.metadata['num_drug_cols']:]
        # for j in range(self.mlp_num_layers):
            # h_gene = getattr(self, f"gene_fc{j+1}")(h_gene)
            # h_drug = getattr(self, f"drug_fc{j+1}")(h_drug)
            # if j != self.mlp_num_layers - 1:
                # h_gene = F.relu(h_gene)
                # h_drug = F.relu(h_drug)

        # batch_output = h_gene + h_drug

        # if compute_loss:
            # loss = self.compute_loss(batch_output, batch['label'])
            # if return_output:
                # return loss, batch_output
            # else:
                # return loss
        # else:
            # return batch_output

    # def compute_loss(self, pred, label):
        # loss = 0.0
        # ret = {}

        # mse_loss = F.mse_loss(pred.flatten(), label)
        # loss += mse_loss
        # ret['mse'] = mse_loss

        # if self.l1_weight:
            # l1_loss = 0.0
            # for parameter in self.parameters():
                # l1_loss += self.l1_weight * parameter.abs().sum()
            # loss += l1_loss
            # ret['l1'] = l1_loss

        # if self.l2_weight:
            # l2_loss = 0.0
            # for parameter in self.parameters():
                # l2_loss += self.l2_weight * parameter.square().sum()
            # loss += l2_loss
            # ret['l2'] = l2_loss

        # ret['total'] = loss
        # return ret

    # def reset_parameters(self):
        # for i in range(self.gsp_num_layers):
            # getattr(self, f"gsp_fc{i+1}").reset_parameters()
        # for i in range(self.mlp_num_layers):
            # getattr(self, f"gene_fc{i+1}").reset_parameters()
            # getattr(self, f"drug_fc{i+1}").reset_parameters()


# class GSPBaselineSimpleGlobal(nn.Module):
    # def __init__(
        # self,
        # gsp_num_comps: int,
        # gsp_num_layers: int,
        # mlp_num_layers: int,
        # mlp_hidden_dim: int,
        # edge_weight_method: str,
        # spectral_matrix_type: str,
        # l1_weight: float,
        # l2_weight: float,
        # kg: Data,
        # metadata: dict,
        # **kwargs
    # ):
        # cls_name = self.__class__.__name__
        # logger.info(f"Initializing {cls_name}")
        # super(GSPBaselineSimpleGlobal, self).__init__()

        # self.gsp_num_comps = gsp_num_comps
        # self.gsp_num_layers = gsp_num_layers
        # self.mlp_num_layers = mlp_num_layers
        # self.mlp_hidden_dim = mlp_hidden_dim
        # self.edge_weight_method = edge_weight_method
        # self.spectral_matrix_type = spectral_matrix_type
        # self.l1_weight = l1_weight
        # self.l2_weight = l2_weight
        # self.kg = kg
        # self.metadata = metadata

        # X_mapping = metadata['X_mapping']
        # # X_mapping = X_mapping[:-metadata['num_drug_cols']]
        # eig_fpath = f"data/plato/kg/eig_{edge_weight_method}_{spectral_matrix_type}_top500.pt"
        # eig_vecs = torch.load(eig_fpath, weights_only=False)['eig_vecs']
        # eig_vecs = eig_vecs[X_mapping, :self.gsp_num_comps]
        # self.register_buffer("eig_vecs", eig_vecs)

        # gene_fc_in_dim = self.gsp_num_comps
        # drug_fc_in_dim = metadata['num_drug_cols']

        # for i in range(self.gsp_num_layers):
            # fc = nn.Linear(in_features=self.gsp_num_comps, out_features=self.gsp_num_comps)
            # setattr(self, f"gsp_fc{i+1}", fc)
        # # self.pool = pyg.nn.global_mean_pool
        # for i in range(self.mlp_num_layers):
            # gene_fc = nn.Linear(
                # in_features=gene_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"gene_fc{i+1}", gene_fc)
            # drug_fc = nn.Linear(
                # in_features=drug_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"drug_fc{i+1}", drug_fc)

        # self.reset_parameters()

    # def forward(self, batch, compute_loss=False, return_output=False):
        # x = batch['input']  # [batch_size, num_cols]

        # # x = x[:, :-self.metadata['num_drug_cols']]
        # x = torch.einsum('bn,nk->bk', x, self.eig_vecs)
        # for i in range(self.gsp_num_layers):
            # fc = getattr(self, f"gsp_fc{i+1}")
            # x = F.relu(fc(x))
        # h_gene = x
        # h_drug = batch['input'][:, -self.metadata['num_drug_cols']:]
        # for j in range(self.mlp_num_layers):
            # h_gene = getattr(self, f"gene_fc{j+1}")(h_gene)
            # h_drug = getattr(self, f"drug_fc{j+1}")(h_drug)
            # if j != self.mlp_num_layers - 1:
                # h_gene = F.relu(h_gene)
                # h_drug = F.relu(h_drug)

        # batch_output = h_gene + h_drug

        # if compute_loss:
            # loss = self.compute_loss(batch_output, batch['label'])
            # if return_output:
                # return loss, batch_output
            # else:
                # return loss
        # else:
            # return batch_output

    # def compute_loss(self, pred, label):
        # loss = 0.0
        # ret = {}

        # mse_loss = F.mse_loss(pred.flatten(), label)
        # loss += mse_loss
        # ret['mse'] = mse_loss

        # if self.l1_weight:
            # l1_loss = 0.0
            # for parameter in self.parameters():
                # l1_loss += self.l1_weight * parameter.abs().sum()
            # loss += l1_loss
            # ret['l1'] = l1_loss

        # if self.l2_weight:
            # l2_loss = 0.0
            # for parameter in self.parameters():
                # l2_loss += self.l2_weight * parameter.square().sum()
            # loss += l2_loss
            # ret['l2'] = l2_loss

        # ret['total'] = loss
        # return ret

    # def reset_parameters(self):
        # for i in range(self.gsp_num_layers):
            # getattr(self, f"gsp_fc{i+1}").reset_parameters()
        # for i in range(self.mlp_num_layers):
            # getattr(self, f"gene_fc{i+1}").reset_parameters()
            # getattr(self, f"drug_fc{i+1}").reset_parameters()


# class GSPBaselineSimpleGlobal(nn.Module):
    # def __init__(
        # self,
        # gsp_num_comps: int,
        # gsp_num_layers: int,
        # mlp_num_layers: int,
        # mlp_hidden_dim: int,
        # edge_weight_method: str,
        # spectral_matrix_type: str,
        # l1_weight: float,
        # l2_weight: float,
        # kg: Data,
        # metadata: dict,
        # **kwargs
    # ):
        # cls_name = self.__class__.__name__
        # logger.info(f"Initializing {cls_name}")
        # super(GSPBaselineSimpleGlobal, self).__init__()

        # self.gsp_num_comps = gsp_num_comps
        # self.gsp_num_layers = gsp_num_layers
        # self.mlp_num_layers = mlp_num_layers
        # self.mlp_hidden_dim = mlp_hidden_dim
        # self.edge_weight_method = edge_weight_method
        # self.spectral_matrix_type = spectral_matrix_type
        # self.l1_weight = l1_weight
        # self.l2_weight = l2_weight
        # self.kg = kg
        # self.metadata = metadata

        # X_mapping = metadata['X_mapping']
        # X_mapping_gene = X_mapping[:-metadata['num_drug_cols']]
        # X_mapping_drug = X_mapping[-metadata['num_drug_cols']:]
        # eig_fpath = f"data/plato/kg/eig_{edge_weight_method}_{spectral_matrix_type}_top500.pt"
        # eig_vecs = torch.load(eig_fpath, weights_only=False)['eig_vecs']
        # eig_vecs_gene = eig_vecs[X_mapping_gene, :self.gsp_num_comps]
        # self.register_buffer("eig_vecs_gene", eig_vecs_gene)

        # kge_drug = kg.x[X_mapping_drug]
        # self.register_buffer("kge_drug", kge_drug)

        # gene_fc_in_dim = self.gsp_num_comps
        # drug_fc_in_dim = kge_drug.shape[1]

        # for i in range(self.mlp_num_layers):
            # gene_fc = nn.Linear(
                # in_features=gene_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"gene_fc{i+1}", gene_fc)
            # drug_fc = nn.Linear(
                # in_features=drug_fc_in_dim if i == 0 else self.mlp_hidden_dim,
                # out_features=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else 1,
            # )
            # setattr(self, f"drug_fc{i+1}", drug_fc)

        # self.reset_parameters()

    # def forward(self, batch, compute_loss=False, return_output=False):
        # x = batch['input']  # [batch_size, num_cols]

        # x_gene = x[:, :-self.metadata['num_drug_cols']]
        # x_gene = torch.einsum('bn,nk->bk', x_gene, self.eig_vecs_gene)
        # h_gene = x_gene
        # x_drug = x[:, -self.metadata['num_drug_cols']:]
        # x_drug = torch.einsum('bn,nk->bk', x_drug, self.kge_drug)
        # h_drug = x_drug
        # for j in range(self.mlp_num_layers):
            # h_gene = getattr(self, f"gene_fc{j+1}")(h_gene)
            # h_drug = getattr(self, f"drug_fc{j+1}")(h_drug)
            # if j != self.mlp_num_layers - 1:
                # h_gene = F.relu(h_gene)
                # h_drug = F.relu(h_drug)

        # batch_output = h_gene + h_drug

        # if compute_loss:
            # loss = self.compute_loss(batch_output, batch['label'])
            # if return_output:
                # return loss, batch_output
            # else:
                # return loss
        # else:
            # return batch_output

    # def compute_loss(self, pred, label):
        # loss = 0.0
        # ret = {}

        # mse_loss = F.mse_loss(pred.flatten(), label)
        # loss += mse_loss
        # ret['mse'] = mse_loss

        # if self.l1_weight:
            # l1_loss = 0.0
            # for parameter in self.parameters():
                # l1_loss += self.l1_weight * parameter.abs().sum()
            # loss += l1_loss
            # ret['l1'] = l1_loss

        # if self.l2_weight:
            # l2_loss = 0.0
            # for parameter in self.parameters():
                # l2_loss += self.l2_weight * parameter.square().sum()
            # loss += l2_loss
            # ret['l2'] = l2_loss

        # ret['total'] = loss
        # return ret

    # def reset_parameters(self):
        # for i in range(self.gsp_num_layers):
            # getattr(self, f"gsp_fc{i+1}").reset_parameters()
        # for i in range(self.mlp_num_layers):
            # getattr(self, f"gene_fc{i+1}").reset_parameters()
            # getattr(self, f"drug_fc{i+1}").reset_parameters()
