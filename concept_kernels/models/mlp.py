import logging
import math
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
import sklearn.cluster

from concept_kernels.models.base import BaseDeepModel
from concept_kernels.datasets.base_dataset import BaseDataset
from concept_kernels.utils.graph_utils import (
    make_symmetric, get_normalized_laplacian, get_topk_eigen,
    get_global_eig_vecs, get_local_eig_vecs
)

logger = logging.getLogger(__name__)


class RealMLPModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input2_dim: int = 0,
        cat_dims: Optional[List[int]] = None,
        num_emb_type: str = "pbld",  # "none", "pbld", "pl", "plr"
        add_front_scale: bool = True,
        p_drop: float = 0.15,
        act: str = "selu",
        hidden_sizes: str = "256,256,256",
        plr_sigma: float = 0.1,
        # Additional hyperparameters not directly in the config but part of the model
        plr_hidden_1: int = 16,
        plr_hidden_2: int = 4,
        embedding_size: int = 8,
    ):
        """
        RealMLP implementation that aligns with parameters in configs/opt_space/realmlp.json

        Args:
            input_dim: Number of continuous features
            output_dim: Number of output dimensions
            cat_dims: List of cardinalities for each categorical feature
            num_emb_type: Type of numerical embeddings. Options:
                - "none": No special numerical embeddings
                - "pbld": Periodic-Bias-Linear-Densenet embeddings
                - "pl": Periodic-Linear embeddings
                - "plr": Periodic-Linear-ReLU embeddings
            add_front_scale: Whether to add scaling layer at beginning
            p_drop: Dropout probability
            act: Activation function ('selu', 'relu', 'mish')
            hidden_sizes: List of hidden layer sizes
            plr_sigma: Initialization standard deviation for PLR embeddings
            plr_hidden_1: Dimensionality of first PLR hidden layer
            plr_hidden_2: Dimensionality of second PLR hidden layer
            embedding_size: Size of categorical embeddings
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cat_dims = cat_dims or []
        self.p_drop = p_drop

        # Set up categorical embeddings if needed
        self.cat_embeddings = None
        if self.cat_dims:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(dim, embedding_size)
                for dim in self.cat_dims
            ])
            # Initialize embeddings
            for emb in self.cat_embeddings:
                nn.init.normal_(emb.weight, 0, 0.1)

            # Add embedded dimensions to total input
            embedded_dim = len(self.cat_dims) * embedding_size
        else:
            embedded_dim = 0

        # Set up numerical embeddings based on num_emb_type
        self.plr_layer = None
        use_plr_embeddings = num_emb_type != "none"

        # Configure PLR parameters based on num_emb_type
        if num_emb_type == "pl":
            use_plr_act = "linear"
            use_densenet = False
            use_cos_bias = False
        elif num_emb_type == "plr":
            use_plr_act = "relu"
            use_densenet = False
            use_cos_bias = False
        elif num_emb_type == "pbld":
            use_plr_act = "linear"
            use_densenet = True
            use_cos_bias = True
        else:  # "none" or invalid values
            use_plr_embeddings = False
            use_plr_act = "linear"
            use_densenet = False
            use_cos_bias = False

        # Create PLR embeddings for numerical features
        if use_plr_embeddings and input_dim > 0:
            self.plr_layer = PLREmbeddingLayer(
                input_dim=input_dim,
                sigma=plr_sigma,
                hidden_1=plr_hidden_1,
                hidden_2=plr_hidden_2,
                use_densenet=use_densenet,
                use_cos_bias=use_cos_bias,
                act_name=use_plr_act
            )
            # PLR increases dimension
            if use_densenet:
                plr_output_dim = input_dim * (plr_hidden_2 + 1)
            else:
                plr_output_dim = input_dim * plr_hidden_2
        else:
            plr_output_dim = input_dim

        # Total input size after embeddings and PLR
        total_input_dim = plr_output_dim + embedded_dim + input2_dim

        # Optional scaling layer at beginning
        self.scale_layer = None
        if add_front_scale and total_input_dim > 0:
            self.scale_layer = ScalingLayer(total_input_dim)

        # Activation function
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()  # default

        # Build MLP layers
        layers = []
        layer_dims = [total_input_dim] + list(map(int, hidden_sizes.split(',')))
        for i in range(len(layer_dims) - 1):
            layers.append(NTPLinear(layer_dims[i], layer_dims[i+1]))
        layers.append(NTPLinear(layer_dims[-1], output_dim, zero_init=True))
        self.layers = nn.Sequential(*layers)

        self.plr_params = None if (self.plr_layer is None) else list(self.plr_layer.parameters())
        self.scale_params = None if (self.scale_layer is None) else list(self.scale_layer.parameters())
        layer_params = list(self.layers.parameters())
        self.weight_params = layer_params[0::2]
        self.bias_params = layer_params[1::2]

    def forward(self, x_num, x_cat=None, x_add=None):
        """
        Forward pass of the RealMLP model

        Args:
            x_num: Tensor of continuous features [batch_size, input_dim]
            x_cat: Tensor of categorical features [batch_size, n_cat_features] or None
            x_add: Tensor of additional features [batch_size, input2_dim]

        Returns:
            Model output (logits for classification, values for regression)
        """
        features = []

        # Process numerical features through PLR if available
        if self.input_dim > 0:
            if self.plr_layer is not None:
                num_features = self.plr_layer(x_num)
            else:
                num_features = x_num
            features.append(num_features)

        # Process categorical features through embeddings
        if x_cat is not None and self.cat_embeddings is not None:
            cat_embedded = []
            for i, embedding in enumerate(self.cat_embeddings):
                cat_embedded.append(embedding(x_cat[:, i].int()))
            if cat_embedded:
                cat_features = torch.cat(cat_embedded, dim=1)
                features.append(cat_features)

        if x_add is not None:
            features.append(x_add)

        # Combine all features
        x = torch.cat(features, dim=1)

        # Apply front scaling if available
        if self.scale_layer is not None:
            x = self.scale_layer(x)

        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation and dropout except for the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
                if self.p_drop > 0 and self.training:
                    x = F.dropout(x, p=self.p_drop, training=self.training)
        return x

    def get_param_groups(self):
        param_groups = [{'params': self.weight_params, 'lr_mult': 1.0},
                        {'params': self.bias_params, 'lr_mult': 0.1}]
        if self.plr_params:
            param_groups.append({'params': self.plr_params, 'lr_mult': 1.0})
        if self.scale_params:
            param_groups.append({'params': self.scale_params, 'lr_mult': 6.0})
        return param_groups


class PLREmbeddingLayer(nn.Module):
    """
    Implements Periodic-Linear-ReLU embeddings for numerical features
    """
    def __init__(
        self,
        input_dim: int,
        sigma: float = 0.1,
        hidden_1: int = 16,
        hidden_2: int = 4,
        use_densenet: bool = True,
        use_cos_bias: bool = True,
        act_name: str = 'linear'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_densenet = use_densenet
        self.use_cos_bias = use_cos_bias

        # First layer params (periodic transformations)
        if use_cos_bias:
            self.weight_1 = nn.Parameter(sigma * torch.randn(input_dim, hidden_1))
            # Use uniform [-pi, pi] initialization for better learning with weight decay
            self.bias_1 = nn.Parameter(np.pi * (-1 + 2 * torch.rand(input_dim, hidden_1)))
        else:
            # Double the features (sin/cos for each)
            self.weight_1 = nn.Parameter(sigma * torch.randn(input_dim, hidden_1 // 2))
            self.bias_1 = None

        # Second layer params
        effective_hidden_1 = hidden_1
        self.weight_2 = nn.Parameter(
            (-1 + 2 * torch.rand(input_dim, effective_hidden_1, hidden_2)) / np.sqrt(effective_hidden_1)
        )
        self.bias_2 = nn.Parameter(
            (-1 + 2 * torch.rand(input_dim, hidden_2)) / np.sqrt(effective_hidden_1)
        )

        # Activation function
        self.act_name = act_name

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, input_dim]

        Returns:
            Embedded features
        """
        # Store original input for potential skip connection
        x_orig = x

        # Reshape to [batch_size, input_dim, 1]
        x = x.unsqueeze(-1)

        # First layer transformation
        if self.use_cos_bias:
            # Cos with bias version: cos(wx + b)
            # x = 2 * torch.pi * torch.matmul(x, self.weight_1.unsqueeze(1))
            x = 2 * torch.pi * x * self.weight_1.unsqueeze(0)
            x = x + self.bias_1.unsqueeze(0)
            x = torch.cos(x)
        else:
            # Sincos version: [sin(wx), cos(wx)]
            # x = 2 * torch.pi * torch.matmul(x, self.weight_1.unsqueeze(1))
            x = 2 * torch.pi * x * self.weight_1.unsqueeze(0)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        # Second layer transformation
        # x = torch.matmul(x, self.weight_2)
        x = torch.einsum('bij,ijk->bik', x, self.weight_2)
        x = x + self.bias_2

        # Apply activation if specified
        if self.act_name == 'relu':
            x = torch.relu(x)
        # 'linear' activation is identity function

        # Reshape to [batch_size, input_dim * hidden_2]
        x = x.reshape(x.shape[0], -1)

        # Add skip connection if using densenet
        if self.use_densenet:
            x = torch.cat([x, x_orig], dim=-1)

        return x


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class RealMLP(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        num_emb_type: str,
        add_front_scale: bool,
        p_drop: float,
        act: str,
        hidden_sizes: str,
        plr_sigma: float,
        ls_eps: float,
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

        self.ls_eps = ls_eps

        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1
        categories = metadata['categories']
        categories = categories if categories else None

        self.model = RealMLPModule(
            input_dim=n_num_features,
            output_dim=pred_dim,
            cat_dims=categories,
            num_emb_type=num_emb_type,
            add_front_scale=add_front_scale,
            p_drop=p_drop,
            act=act,
            hidden_sizes=hidden_sizes,
            plr_sigma=plr_sigma,
        )

    def forward(self, batch, compute_loss=False, return_output=False):
        x = self.model(batch['X_num'], batch['X_cat'])
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def get_param_groups(self):
        return self.model.get_param_groups()

    def compute_loss(self, pred, label):
        loss = 0.0
        ret = {}

        if self.task_type == 'regression':
            mse_loss = F.mse_loss(pred.flatten(), label)
            loss += mse_loss
            ret['mse'] = mse_loss
        elif self.task_type == 'binclass':
            label = label.float() * (1.0 - self.ls_eps) + 0.5 * self.ls_eps
            xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label)
            # xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label.float())
            loss += xent_loss
            ret['xent'] = xent_loss
        elif self.task_type == 'multiclass':
            xent_loss = F.cross_entropy(pred, label, label_smoothing=self.ls_eps)
            loss += xent_loss
            ret['xent'] = xent_loss

        ret['total'] = loss
        return ret


class MLPSmooth(BaseDeepModel):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        input_smooth_mode: str,
        input_smooth_lambda: float,
        num_emb_type: str,
        add_front_scale: bool,
        p_drop: float,
        act: str,
        hidden_sizes: str,
        plr_sigma: float,
        ls_eps: float,
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

        input_filter = self.get_smooth_filter(kg.x, input_smooth_mode, input_smooth_lambda)
        self.register_buffer('input_filter', input_filter)
        self.ls_eps = ls_eps

        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1
        categories = metadata['categories']
        categories = categories if categories else None

        input_dim = n_num_features
        if categories:
            if self.cat_enc_policy in ['target']:
                input_dim += len(categories)
            else:
                input_dim += sum(categories)

        self.model = RealMLPModule(
            input_dim=input_dim,
            output_dim=pred_dim,
            cat_dims=None,
            num_emb_type=num_emb_type,
            add_front_scale=add_front_scale,
            p_drop=p_drop,
            act=act,
            hidden_sizes=hidden_sizes,
            plr_sigma=plr_sigma,
        )

    def get_smooth_filter(self, col_embed, input_smooth_mode, input_smooth_lambda):
        n = col_embed.shape[0]
        K = col_embed @ col_embed.T
        if input_smooth_mode == 'kernel_conv':
            input_filter = K / K.sum(dim=1, keepdims=True)
        elif input_smooth_mode == 'kernel_conv2':
            d = K.sum(dim=0)
            d_sqrt_inv = torch.where(d > 1e-8, d**-0.5, torch.zeros_like(d))
            input_filter = d_sqrt_inv.unsqueeze(1) * K * d_sqrt_inv.unsqueeze(0)
        elif input_smooth_mode == 'rkhs_norm':
            K_inv = torch.linalg.inv(K + input_smooth_lambda * torch.eye(n))
            input_filter = K @ K_inv
        elif input_smooth_mode == 'laplacian':
            d = K.sum(dim=0)
            d_sqrt_inv = torch.where(d > 1e-8, d**-0.5, torch.zeros_like(d))
            K_bar = d_sqrt_inv.unsqueeze(1) * K * d_sqrt_inv.unsqueeze(0)
            temp =  (1 + input_smooth_lambda) * torch.eye(n) - input_smooth_lambda * K_bar
            input_filter = torch.linalg.inv(temp)
        else:
            raise ValueError(f"Wrong smooth mode: {input_smooth_mode}")
        return input_filter

    def forward(self, batch, compute_loss=False, return_output=False):
        x = [batch['X_num'], batch['X_cat']]
        x = torch.cat([t for t in x if t is not None], dim=1)
        x = x @ self.input_filter.T

        x = self.model(x, None)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def get_param_groups(self):
        return self.model.get_param_groups()

    def compute_loss(self, pred, label):
        loss = 0.0
        ret = {}

        if self.task_type == 'regression':
            mse_loss = F.mse_loss(pred.flatten(), label)
            loss += mse_loss
            ret['mse'] = mse_loss
        elif self.task_type == 'binclass':
            label = label.float() * (1.0 - self.ls_eps) + 0.5 * self.ls_eps
            xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label)
            # xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label.float())
            loss += xent_loss
            ret['xent'] = xent_loss
        elif self.task_type == 'multiclass':
            xent_loss = F.cross_entropy(pred, label, label_smoothing=self.ls_eps)
            loss += xent_loss
            ret['xent'] = xent_loss

        ret['total'] = loss
        return ret


class MLPGSP(BaseDeepModel):
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
        num_emb_type: str,
        add_front_scale: bool,
        p_drop: float,
        act: str,
        hidden_sizes: str,
        plr_sigma: float,
        ls_eps: float,
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
        self.ls_eps = ls_eps
        self.kg = kg
        self.metadata = metadata

        feature_dim = self.n_num_features + self.n_cat_features
        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1
        categories = metadata['categories']
        self.categories = categories if categories else None

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

        self.mlp = RealMLPModule(
            input_dim=n_num_features,
            input2_dim=eig_vecs.shape[1],
            output_dim=pred_dim,
            cat_dims=categories,
            num_emb_type=num_emb_type,
            add_front_scale=add_front_scale,
            p_drop=p_drop,
            act=act,
            hidden_sizes=hidden_sizes,
            plr_sigma=plr_sigma,
        )

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
        x_num, x_cat, x_cat2 = batch['X_num'], batch['X_cat'], None
        x = [x_num, x_cat]
        x = torch.cat([t for t in x if t is not None], dim=1)
        x = torch.einsum('bn,nk->bk', x, self.eig_vecs)

        if x_cat is not None:
            x_cat2 = []
            co = 0
            for nc in self.categories:
                x_cat2.append(torch.argmax(x_cat[:, co:co+nc], dim=1))
                co += nc
            x_cat2 = torch.stack(x_cat2, dim=1)
        x = self.mlp(x_num, x_cat2, x)
        batch_output = x

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def get_param_groups(self):
        return self.mlp.get_param_groups()

    def compute_loss(self, pred, label):
        loss = 0.0
        ret = {}

        if self.task_type == 'regression':
            mse_loss = F.mse_loss(pred.flatten(), label)
            loss += mse_loss
            ret['mse'] = mse_loss
        elif self.task_type == 'binclass':
            label = label.float() * (1.0 - self.ls_eps) + 0.5 * self.ls_eps
            xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label)
            # xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label.float())
            loss += xent_loss
            ret['xent'] = xent_loss
        elif self.task_type == 'multiclass':
            xent_loss = F.cross_entropy(pred, label, label_smoothing=self.ls_eps)
            loss += xent_loss
            ret['xent'] = xent_loss

        ret['total'] = loss
        return ret
