import logging
import os
import random

import torch
import xgboost as xgb
import torch_geometric as pyg
from torch_geometric.data import Data

from concept_kernels.utils.graph_utils import (
    make_symmetric, get_normalized_laplacian, get_topk_eigen
)

logger = logging.getLogger(__name__)


class XGBGSPRegressor():
    def __init__(
        self,
        kg: Data,
        metadata: dict,
        **kwargs
    ):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        super(XGBGSPRegressor, self).__init__()

        self.gsp_num_comps = kwargs['gsp_num_comps']
        self.edge_weight_method = kwargs['edge_weight_method']
        self.kg = kg
        self.metadata = metadata

        # Compute the eigen vectors
        eig_vec_cache_path = f"/tmp/eig_{self.edge_weight_method}_{metadata['num_gene_cols']}.pt"
        if os.path.exists(eig_vec_cache_path):
            eig_vecs = torch.load(eig_vec_cache_path, weights_only=False)
        else:
            X_mapping = metadata['X_mapping']
            X_mapping = X_mapping[:-metadata['num_drug_cols']]
            V = kg.x[X_mapping]
            edge_index, edge_attr = pyg.utils.subgraph(
                X_mapping, kg.edge_index, kg.edge_attr, relabel_nodes=True)
            if self.edge_weight_method == 'uniform':
                edge_weight = None
            elif self.edge_weight_method == 'cosine':
                from_V, to_V = V[edge_index[0]], V[edge_index[1]]
                edge_weight = (from_V*to_V).sum(dim=1) / from_V.norm(dim=1) / to_V.norm(dim=1)
            elif self.edge_weight_method == 'score':
                from_V, to_V = V[edge_index[0]], V[edge_index[1]]
                edge_weight = (from_V*kg.relation_embedding[edge_attr]*to_V).sum(dim=1)
                edge_weight = torch.log(1.0 + torch.clip(edge_weight, min=0.0))
            edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
            num_nodes = V.shape[0]
            laplacian = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
            eig_vals, eig_vecs = get_topk_eigen(laplacian, 1000)
            torch.save(eig_vecs, eig_vec_cache_path)
        self.eig_vecs = eig_vecs[:, :self.gsp_num_comps]

        xgb_conf = kwargs.copy()
        del xgb_conf['gsp_num_comps']
        del xgb_conf['edge_weight_method']

        self.gene_dim = metadata['num_gene_cols']
        self.model = xgb.XGBRegressor(**xgb_conf)
        self._callbacks = None

    # @property
    # def callbacks(self):
        # return self._callbacks

    # @callbacks.setter
    # def callbacks(self, value):
        # self._callbacks = value
        # self.gene_model.callbacks = value
        # self.drug_model.callbacks = value

    def convert_X(self, X):
        X_gene = X[:, :self.gene_dim] @ self.eig_vecs
        X_drug = X[:, self.gene_dim:]
        return torch.cat([X_gene, X_drug], dim=1)

    def fit(self, X_train, y_train, eval_set, verbose):
        X_train_cvt = self.convert_X(X_train)
        eval_set_cvt = [(self.convert_X(X), y) for X, y in eval_set]
        self.model.fit(X_train_cvt, y_train, eval_set=eval_set_cvt, verbose=True)
        return self

    def predict(self, X):
        X_cvt = self.convert_X(X)
        return self.model.predict(X_cvt)

    def save_model(self, ckpt_fpath):
        self.model.save_model(ckpt_fpath)

    def load_model(self, ckpt_fpath):
        self.model.load_model(ckpt_fpath)

    def get_total_params(self):
        tree_dump = self.model.get_booster().get_dump()
        splits = sum(line.count('[') for tree in tree_dump for line in tree.split('\n'))
        leaves = sum(line.count('leaf=') for tree in tree_dump for line in tree.split('\n'))
        total_params = splits * 2 + leaves
        return total_params


class XGBGSPRegressorSeq():
    def __init__(
        self,
        kg: Data,
        metadata: dict,
        **kwargs
    ):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        super(XGBGSPRegressorSeq, self).__init__()

        self.gsp_num_comps = kwargs['gsp_num_comps']
        self.edge_weight_method = kwargs['edge_weight_method']
        self.train_order = kwargs['train_order']
        self.kg = kg
        self.metadata = metadata

        # Compute the eigen vectors
        eig_vec_cache_path = f"/tmp/eig_{self.edge_weight_method}_{metadata['num_gene_cols']}.pt"
        if os.path.exists(eig_vec_cache_path):
            eig_vecs = torch.load(eig_vec_cache_path, weights_only=False)
        else:
            X_mapping = metadata['X_mapping']
            X_mapping = X_mapping[:-metadata['num_drug_cols']]
            V = kg.x[X_mapping]
            edge_index, edge_attr = pyg.utils.subgraph(
                X_mapping, kg.edge_index, kg.edge_attr, relabel_nodes=True)
            if self.edge_weight_method == 'uniform':
                edge_weight = None
            elif self.edge_weight_method == 'cosine':
                from_V, to_V = V[edge_index[0]], V[edge_index[1]]
                edge_weight = (from_V*to_V).sum(dim=1) / from_V.norm(dim=1) / to_V.norm(dim=1)
            elif self.edge_weight_method == 'score':
                from_V, to_V = V[edge_index[0]], V[edge_index[1]]
                edge_weight = (from_V*kg.relation_embedding[edge_attr]*to_V).sum(dim=1)
                edge_weight = torch.log(1.0 + torch.clip(edge_weight, min=0.0))
            edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
            num_nodes = V.shape[0]
            laplacian = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
            eig_vals, eig_vecs = get_topk_eigen(laplacian, 1000)
            torch.save(eig_vecs, eig_vec_cache_path)
        self.eig_vecs = eig_vecs[:, :self.gsp_num_comps]

        xgb_conf = kwargs.copy()
        del xgb_conf['gsp_num_comps']
        del xgb_conf['edge_weight_method']
        del xgb_conf['train_order']

        self.gene_dim = metadata['num_gene_cols']
        self.models = {feat: xgb.XGBRegressor(**xgb_conf) for feat in self.train_order}
        self._callbacks = None

    # @property
    # def callbacks(self):
        # return self._callbacks

    # @callbacks.setter
    # def callbacks(self, value):
        # self._callbacks = value
        # self.gene_model.callbacks = value
        # self.drug_model.callbacks = value

    def split_X(self, X):
        X_gene = X[:, :self.gene_dim] @ self.eig_vecs
        X_drug = X[:, self.gene_dim:]
        ret = {'gene': X_gene, 'drug': X_drug}
        return ret
    
    def get_random_idxs(self, n_val):
        n_model = len(self.train_order)
        idx_cnts = [(n_val + i) // n_model for i in range(n_model)]
        assert(n_val == sum(idx_cnts))

        val_idx = list(range(n_val))
        random.shuffle(val_idx)
        ret = []
        cnt = 0
        for i in range(n_model):
            ret.append(val_idx[cnt:cnt+idx_cnts[i]])
            cnt += idx_cnts[i];
        return ret

    def fit(self, X_train, y_train, eval_set, verbose):
        X_train_split = self.split_X(X_train)
        eval_set_split = {feat: [] for feat in self.train_order}
        eval_split_idxs = {feat: [] for feat in self.train_order}
        for X, y in eval_set:
            X_split = self.split_X(X)
            temp_idxs = self.get_random_idxs(len(X))
            for i, feat in enumerate(self.train_order):
                eval_set_split[feat].append((X_split[feat], y))
                eval_split_idxs[feat].append(temp_idxs[i])

        for i, feat in enumerate(self.train_order):
            model = self.models[feat]
            model.fit(X_train_split[feat], y_train,
                      eval_set=[(X[idxs], y[idxs]) for (X, y), idxs in
                                zip(eval_set_split[feat], eval_split_idxs[feat])],
                      verbose=True)
            y_train = y_train - model.predict(X_train_split[feat])
            for i in range(len(eval_set)):
                pred = model.predict(eval_set_split[feat][i][0])
                for feat2 in self.train_order:
                    X, y = eval_set_split[feat2][i]
                    eval_set_split[feat2][i] = (X, y - pred)

        return self

    def predict(self, X):
        X_split = self.split_X(X)
        preds = []
        for feat in self.train_order:
            preds.append(self.models[feat].predict(X_split[feat]))
        return sum(preds)

    def get_ckpt_paths(self, ckpt_fpath):
        before_ext = os.path.splitext(ckpt_fpath)[0]
        ret = {feat: before_ext + f"_{feat}.ubj" for feat in self.train_order}
        return ret

    def save_model(self, ckpt_fpath):
        ckpt_fpaths = self.get_ckpt_paths(ckpt_fpath)
        for feat, fpath in ckpt_fpaths.items():
            self.models[feat].save_model(fpath)

    def load_model(self, ckpt_fpath):
        ckpt_fpaths = self.get_ckpt_paths(ckpt_fpath)
        for feat, fpath in ckpt_fpaths.items():
            self.models[feat].load_model(fpath)

    def checkpoint_exists(self, ckpt_fpath):
        ckpt_fpaths = self.get_ckpt_paths(ckpt_fpath)
        return all([os.path.exists(fpath) for _, fpath in ckpt_fpaths.items()])

    def get_total_params(self):
        total_params = 0
        for _, model in self.models.items():
            tree_dump = model.get_booster().get_dump()
            splits = sum(line.count('[') for tree in tree_dump for line in tree.split('\n'))
            leaves = sum(line.count('leaf=') for tree in tree_dump for line in tree.split('\n'))
            total_params += splits * 2 + leaves
        return total_params
