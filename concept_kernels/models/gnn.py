import logging
import math
import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

from concept_kernels.models.base import BaseDeepModel
from concept_kernels.models.ftt import Tokenizer

logger = logging.getLogger(__name__)


class GNNBaseline(BaseDeepModel):
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
        # edge_binarize_mode: str,
        edge_active_ratio: float,
        # edge_active_ratio2: float,
        mlp_num_layers: int,
        mlp_hidden_dim: int,
        dropout: float,
        kg: Data,
        metadata: dict,
        concept_proj_dim: int=-1,
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
        # Define GCN and output layer (single number)
        self.input_embed_dim = input_embed_dim
        self.conv_layer_type = conv_layer_type
        self.conv_num_layers = conv_num_layers
        self.conv_hidden_dim = conv_hidden_dim
        self.concept_proj_dim = concept_proj_dim
        self.gat_num_heads = gat_num_heads
        self.conv_pool_method = conv_pool_method
        # self.edge_binarize_mode = edge_binarize_mode
        self.edge_active_ratio = edge_active_ratio
        # self.edge_active_ratio2 = edge_active_ratio2
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.kg = kg
        self.metadata = metadata

        pred_dim = self.num_classes if self.task_type == 'multiclass' else 1

        # Select edge_active_ratio of the pos/neg edges with large magnitude
        V, edge_index, edge_weight = kg.x, kg.edge_index, kg.edge_attr
        n_num_features = self.metadata['n_num_features']
        n_cat_features = self.metadata['n_cat_features']
        categories = self.metadata.get('categories', None)
        edge_mask = self.threshold_edge_mask(edge_weight, edge_active_ratio)
        # if self.edge_binarize_mode == 'all' or n_cat_features == 0:
            # edge_mask = self.threshold_edge_mask(edge_weight, edge_active_ratio)
        # elif self.edge_binarize_mode == 'block':
            # out_active_ratio = edge_active_ratio
            # in_active_ratio = edge_active_ratio2 if (edge_active_ratio2) else out_active_ratio
            # categories = self.metadata['categories']
            # cat_offset = n_num_features
            # edge_mask = self.threshold_edge_mask_block(
                # edge_index, edge_weight, categories, cat_offset, in_active_ratio, out_active_ratio
            # )
        edge_index = edge_index[:, edge_mask]
        self.register_buffer("V", V)
        self.register_buffer("edge_index", edge_index)

        self.tokenizer = Tokenizer(n_num_features, categories, self.input_embed_dim, True)

        # Input embedding
        # self.input_embed = nn.Parameter(torch.zeros(V.shape[0], self.input_embed_dim))
        # self.embed_fc = nn.Linear(V.shape[1], self.input_embed_dim)

        for i in range(self.conv_num_layers):
            if self.conv_layer_type == 'GCNConv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim
                out_dim = self.conv_hidden_dim
                conv = pyg.nn.GCNConv(in_channels=in_dim, out_channels=out_dim)
            elif self.conv_layer_type == 'GATConv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim * self.gat_num_heads
                out_dim = self.conv_hidden_dim
                conv = pyg.nn.GATConv(in_channels=in_dim, out_channels=out_dim, heads=self.gat_num_heads)
            elif self.conv_layer_type == 'GATv2Conv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim * self.gat_num_heads
                out_dim = self.conv_hidden_dim
                conv = pyg.nn.GATv2Conv(in_channels=in_dim, out_channels=out_dim, heads=self.gat_num_heads,
                                        add_self_loops=False, residual=True)
            elif self.conv_layer_type == 'GATConceptConv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim * self.gat_num_heads
                out_dim = self.conv_hidden_dim
                conv = GATConceptConv(in_channels=in_dim, out_channels=out_dim, heads=self.gat_num_heads,
                                      concept_embed_dim=V.shape[1], concept_proj_dim=self.concept_proj_dim,
                                      add_self_loops=False, residual=True)
            elif self.conv_layer_type == 'GATConcept2Conv':
                in_dim = self.input_embed_dim if i == 0 else self.conv_hidden_dim * self.gat_num_heads
                out_dim = self.conv_hidden_dim
                conv = GATConcept2Conv(in_channels=in_dim, out_channels=out_dim, heads=self.gat_num_heads,
                                       concept_embed_dim=V.shape[1], add_self_loops=False, residual=True)
            setattr(self, f"conv{i+1}", conv)
        layers = []
        for i in range(self.mlp_num_layers):
            if self.conv_layer_type == 'GCNConv':
                in_dim=self.conv_hidden_dim if i == 0 else self.mlp_hidden_dim
            elif self.conv_layer_type == 'GATConv':
                in_dim=self.conv_hidden_dim * self.gat_num_heads if i == 0 else self.mlp_hidden_dim
            elif self.conv_layer_type == 'GATv2Conv':
                in_dim=self.conv_hidden_dim * self.gat_num_heads if i == 0 else self.mlp_hidden_dim
            elif self.conv_layer_type == 'GATConceptConv':
                in_dim=self.conv_hidden_dim * self.gat_num_heads if i == 0 else self.mlp_hidden_dim
            elif self.conv_layer_type == 'GATConcept2Conv':
                in_dim=self.conv_hidden_dim * self.gat_num_heads if i == 0 else self.mlp_hidden_dim
            out_dim=self.mlp_hidden_dim if i != self.mlp_num_layers-1 else pred_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i != self.mlp_num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def threshold_edge_mask(self, edge_weight, active_ratio):
        w_pos = sorted(edge_weight[edge_weight > 0.0].tolist(), reverse=True)
        w_neg = sorted(edge_weight[edge_weight < 0.0].tolist())
        x_pos = w_pos[math.ceil(len(w_pos)*active_ratio)-1]
        x_neg = w_neg[math.ceil(len(w_neg)*active_ratio)-1] if w_neg else -1e8
        edge_mask = (edge_weight >= x_pos) | (edge_weight <= x_neg)
        return edge_mask

    def threshold_edge_mask_block(self, edge_index, edge_weight,
                                  categories, cat_offset,
                                  in_active_ratio, off_active_ratio):
        block_mask = torch.zeros_like(edge_weight, dtype=torch.bool)
        row, col = edge_index[0, :], edge_index[1, :]
        offset = cat_offset
        for nc in categories:
            block_mask |= (row >= offset) & (row < offset+nc) & (col >= offset) & (col < offset+nc)
            offset += nc
        in_edge_mask = self.threshold_edge_mask(edge_weight[block_mask], in_active_ratio)
        out_edge_mask = self.threshold_edge_mask(edge_weight[~block_mask], off_active_ratio)
        edge_mask = torch.zeros_like(edge_weight, dtype=torch.bool)
        edge_mask[block_mask] = in_edge_mask
        edge_mask[~block_mask] = out_edge_mask
        return edge_mask

    def forward(self, batch, compute_loss=False, return_output=False):
        # x = [batch['X_num'], batch['X_cat']]
        # x = torch.cat([t for t in x if t is not None], dim=1)

        # x = x.unsqueeze(2) * self.input_embed.unsqueeze(0)
        # x = x.unsqueeze(2) * self.embed_fc(self.V).unsqueeze(0)
        # x = x.unsqueeze(2) * self.V.unsqueeze(0)
        X_num, X_cat = batch['X_num'], batch['X_cat']
        if X_cat is not None:
            X_cat = X_cat.int()
        x = self.tokenizer(X_num, X_cat)[:, 1:]
        x = x.contiguous()

        B, D = x.shape[:2]
        E = self.edge_index.shape[1]
        x = x.view(B * D, -1)
        edge_index = self.edge_index.repeat(1, B)
        edge_index += torch.arange(B, device=edge_index.device).repeat_interleave(E) * D
        if self.conv_layer_type in ['GATConceptConv', 'GATConcept2Conv']:
            ce = self.V.repeat(B, 1)
        for j in range(self.conv_num_layers):
            conv = getattr(self, f"conv{j+1}")
            if self.conv_layer_type in ['GATConceptConv', 'GATConcept2Conv']:
                x = conv(x, edge_index=edge_index, ce=ce)
            else:
                x = conv(x, edge_index=edge_index)
            x = F.relu(x)
        x = x.view(B, D, -1)
        if self.conv_pool_method == 'max':
            x, _ = x.max(dim=1)
        elif self.conv_pool_method == 'mean':
            x = x.mean(dim=1)

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

    def reset_parameters(self):
        if hasattr(self, 'input_feature'):
            nn.init.normal_(self.input_embed, std=1.0/self.input_embed_dim**0.5)
        for i in range(self.conv_num_layers):
            getattr(self, f"conv{i+1}").reset_parameters()


class GATConceptConv(MessagePassing):
    """Modified from GATv2Conv of torch-geometric"""
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        concept_embed_dim: int,
        concept_proj_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concept_embed_dim = concept_embed_dim
        self.concept_proj_dim = concept_proj_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.lin_concept_i = Linear(concept_embed_dim, heads * concept_proj_dim,
                                    bias=bias, weight_initializer='glorot')
        self.lin_concept_j = Linear(concept_embed_dim, heads * concept_proj_dim,
                                    bias=bias, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)
        self.lin_concept_i.reset_parameters()
        self.lin_concept_j.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        ce: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C, C2 = self.heads, self.out_channels, self.concept_proj_dim

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        assert isinstance(x, Tensor) and x.dim() == 2
        if self.res is not None:
            res = self.res(x)

        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, _ = remove_self_loops(
                    edge_index, edge_attr=None)
                edge_index, _ = add_self_loops(
                    edge_index, edge_attr=None, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        ce_i = self.lin_concept_i(ce).view(-1, H, C2)
        ce_j = self.lin_concept_j(ce).view(-1, H, C2)
        ce_i, ce_j = ce_i[edge_index[0], :, :], ce_j[edge_index[1], :, :]
        ce_alpha = (ce_i * ce_j).sum(dim=-1)

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=ce_alpha)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        assert len(edge_attr.shape) == 2 and edge_attr.shape == x.shape[:2]

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha += edge_attr
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GATConcept2Conv(MessagePassing):
    """Modified from GATv2Conv of torch-geometric"""
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        concept_embed_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concept_embed_dim = concept_embed_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.lin_concept_i = Linear(concept_embed_dim, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        self.lin_concept_j = Linear(concept_embed_dim, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)
        self.lin_concept_i.reset_parameters()
        self.lin_concept_j.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        ce: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        assert isinstance(x, Tensor) and x.dim() == 2
        if self.res is not None:
            res = self.res(x)

        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, _ = remove_self_loops(
                    edge_index, edge_attr=None)
                edge_index, _ = add_self_loops(
                    edge_index, edge_attr=None, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        ce_i = self.lin_concept_i(ce).view(-1, H, C)
        ce_j = self.lin_concept_j(ce).view(-1, H, C)
        ce_i, ce_j = ce_i[edge_index[0], :, :], ce_j[edge_index[1], :, :]
        ce2 = ce_i + ce_j

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=ce2)

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j
        assert edge_attr.shape == x.shape
        x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
