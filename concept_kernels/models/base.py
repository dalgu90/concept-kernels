import copy
import pickle
import warnings

from category_encoders import (
    BinaryEncoder,
    CatBoostEncoder,
    TargetEncoder,
)
import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from concept_kernels.datasets.base_dataset import BaseDataset


class SmoothedStandardBinaryScaler(BaseEstimator, TransformerMixin):
    def __init__(self, prior_count=1.0):
        self.prior_count = prior_count

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_samples_, self.n_features_ = X.shape

        # Compute empirical means and variances
        self.empirical_mean_ = np.mean(X, axis=0)

        # Incorporate prior: prior_count zeros and prior_count ones
        prior_ones = self.prior_count
        prior_zeros = self.prior_count
        prior_total = prior_ones + prior_zeros

        # Prior mean and variance for binary prior
        prior_mean = prior_ones / prior_total  # always 0.5 for symmetric prior

        # Compute smoothed mean
        self.smoothed_mean_ = (
            (self.empirical_mean_ * self.n_samples_ + prior_mean * prior_total)
            / (self.n_samples_ + prior_total)
        )

        # Compute smoothed variance
        self.smoothed_var_ = self.smoothed_mean_ * (1.0 - self.smoothed_mean_)

        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self.smoothed_mean_) / np.sqrt(self.smoothed_var_)


class BaseModel():
    def __init__(
        self,
        num_imp_policy: str = 'mean',
        num_enc_policy: str = 'none',
        num_norm_policy: str = 'standard',
        cat_imp_policy: str = 'new',
        cat_enc_policy: str = 'ordinal',
        **kwargs,
    ):
        self.num_imp_policy = num_imp_policy
        self.num_enc_policy = num_enc_policy
        self.num_norm_policy = num_norm_policy
        self.cat_imp_policy = cat_imp_policy
        self.cat_enc_policy = cat_enc_policy

        self.num_imputer = None
        self.num_encoder = None
        self.num_normalizer = None
        self.cat_imputer = None
        self.cat_ord_encoder = None
        self.cat_ord_mode = None
        self.cat_encoder = None
        self.label_encoder = None

    def data_preproc(self, dataset: BaseDataset):
        if dataset.is_preprocessed:
            class_name = {self.__class__.__name__}
            warnings.warn(f"{class_name}.data_preproc is called after preprocessing")

        self.preproc_label(dataset)
        self.preproc_num_imp(dataset)
        self.preproc_num_enc(dataset)
        self.preproc_num_norm(dataset)
        self.preproc_cat_imp(dataset)
        self.preproc_cat_enc(dataset)
        dataset.is_preprocessed = True

    def preproc_num_imp(self, dataset: BaseDataset):
        if dataset.X_num is None or self.num_imp_policy == 'none':
            return

        # Numerical data imputation ('mean' or 'median')
        assert self.num_imputer is not None or dataset.split == 'train'
        if self.num_imputer is None:
            self.num_imputer = SimpleImputer(
                missing_values=np.nan,
                strategy=self.num_imp_policy
            ).fit(dataset.X_num)
        dataset.X_num = self.num_imputer.transform(dataset.X_num)

    def preproc_num_enc(self, dataset: BaseDataset):
        pass

    def preproc_num_norm(self, dataset: BaseDataset):
        if dataset.X_num is None or self.num_norm_policy == 'none':
            return

        # Numurical data normalization ('standard' only)
        assert self.num_normalizer is not None or dataset.split == 'train'
        if self.num_normalizer is None:
            if self.num_norm_policy == 'standard':
                self.num_normalizer = StandardScaler()
            else:
                raise ValueError(f"Wrong num norm policy: {self.num_norm_policy}")
            self.num_normalizer.fit(dataset.X_num)
        dataset.X_num = self.num_normalizer.transform(dataset.X_num)

    def preproc_cat_imp(self, dataset: BaseDataset):
        if dataset.X_cat is None or self.cat_imp_policy == 'none':
            return

        # Categorical data imputation ('new' or 'most_frequent')
        assert self.cat_imputer is not None or dataset.split == 'train'
        if self.cat_imputer is None:
            if self.cat_imp_policy == 'new':
                self.cat_imputer = SimpleImputer(
                    strategy='constant',
                    fill_value='~new~'
                )
            elif self.cat_imp_policy == 'most_frequent':
                self.cat_imputer = SimpleImputer(
                    strategy='most_frequent'
                )
            else:
                raise ValueError(f"Wrong cat imp policy: {self.cat_imp_policy}")
            # if dataset.X_cat.dtype.kind == 'U':
                # dataset.X_cat = dataset.X_cat.astype(object)
            dataset.X_cat = dataset.X_cat.astype(object)
            self.cat_imputer.fit(dataset.X_cat)
        dataset.X_cat = self.cat_imputer.transform(dataset.X_cat)

    def preproc_cat_enc(self, dataset: BaseDataset):
        if dataset.X_cat is None or self.cat_enc_policy == 'none':
            return

        # Ordinal encoding + handling unseen categories
        assert self.cat_ord_encoder is not None or dataset.split == 'train'
        unknown_value = np.iinfo('int64').max - 3
        if self.cat_ord_encoder is None:
            self.cat_ord_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=unknown_value,
                dtype='int64'
            ).fit(dataset.X_cat)
        dataset.X_cat = self.cat_ord_encoder.transform(dataset.X_cat)
        if self.cat_ord_mode is None:
            assert dataset.split == 'train'
            self.cat_ord_mode = scipy.stats.mode(dataset.X_cat, axis=0).mode
        dataset.X_cat = np.where(
            dataset.X_cat == unknown_value,
            self.cat_ord_mode[np.newaxis],  # If this does not work, try below
            # np.tile(self.cat_ord_mode, (dataset.X_cat.shape[0], 1)),
            dataset.X_cat
        )

        # Categorical encoding of choice
        if self.cat_enc_policy in ['ordinal', 'indices']:  # No cat_encoder needed
            dataset.X_cat = dataset.X_cat.astype('float32')
            return

        assert self.cat_encoder is not None or dataset.split == 'train'
        if self.cat_enc_policy == 'one_hot':
            if self.cat_encoder is None:
                self.cat_encoder = OneHotEncoder(
                    handle_unknown='ignore', sparse_output=False, dtype='float32'
                ).fit(dataset.X_cat)
            dataset.X_cat = self.cat_encoder.transform(dataset.X_cat)
        elif self.cat_enc_policy == 'standard':
            if self.cat_encoder is None:
                self.cat_encoder = StandardScaler().fit(dataset.X_cat)
            dataset.X_cat = self.cat_encoder.transform(dataset.X_cat)
        elif self.cat_enc_policy == 'standard_smooth':
            if self.cat_encoder is None:
                self.cat_encoder = SmoothedStandardBinaryScaler(30).fit(dataset.X_cat)
            dataset.X_cat = self.cat_encoder.transform(dataset.X_cat)
        elif self.cat_enc_policy == 'target':
            if self.cat_encoder is None:
                y = (dataset.y - dataset.y.mean()) / dataset.y.std()
                self.cat_encoder = TargetEncoder(return_df=False).fit(
                    dataset.X_cat.astype(str), y
                )
            dataset.X_cat = self.cat_encoder.transform(dataset.X_cat.astype(str))
        elif self.cat_enc_policy == 'catboost':
            if self.cat_encoder is None:
                self.cat_encoder = TargetEncoder().fit(
                    dataset.X_cat.astype(str), dataset.y
                )
            dataset.X_cat = self.cat_encoder.transform(dataset.X_cat.astype(str))
        else:
            raise ValueError(f"Wrong cat enc policy: {self.cat_enc_policy}")

    def preproc_label(self, dataset: BaseDataset):
        if 'task_type' not in dataset.metadata:
            raise ValueError('Dataset does not have task type')
        task_type = dataset.metadata['task_type']
        if task_type == 'regression':
            if self.label_encoder is None:
                self.label_encoder = StandardScaler().fit(dataset.y.reshape(-1, 1))
            dataset.y = self.label_encoder.transform(dataset.y.reshape(-1, 1)).flatten().astype('float32')
        elif task_type == 'binclass' or task_type == 'multiclass':
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder().fit(dataset.y)
            dataset.y = self.label_encoder.transform(dataset.y)
        else:
            raise ValueError(f"Cannot recognize task type: {task_type}")

    def dumps_encoders(self):
        dumps = {
            'num_imputer': pickle.dumps(self.num_imputer),
            'num_encoder': pickle.dumps(self.num_encoder),
            'num_normalizer': pickle.dumps(self.num_normalizer),
            'cat_imputer': pickle.dumps(self.cat_imputer),
            'cat_ord_encoder': pickle.dumps(self.cat_ord_encoder),
            'cat_ord_mode': pickle.dumps(self.cat_ord_mode),
            'cat_encoder': pickle.dumps(self.cat_encoder),
            'label_encoder': pickle.dumps(self.label_encoder),
        }
        return dumps

    def loads_encoders(self, dumps):
        self.num_imputer = pickle.loads(dumps['num_imputer'])
        self.num_encoder = pickle.loads(dumps['num_encoder'])
        self.num_normalizer = pickle.loads(dumps['num_normalizer'])
        self.cat_imputer = pickle.loads(dumps['cat_imputer'])
        self.cat_ord_encoder = pickle.loads(dumps['cat_ord_encoder'])
        self.cat_ord_mode = pickle.loads(dumps['cat_ord_mode'])
        self.cat_encoder = pickle.loads(dumps['cat_encoder'])
        self.label_encoder = pickle.loads(dumps['label_encoder'])


class BaseDeepModel(BaseModel, torch.nn.Module):
    def __init__(
        self,
        task_type: str,
        num_classes: int,
        n_num_features: int,
        n_cat_features: int,
        kg: Data,
        metadata: dict,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        **kwargs,
    ):
        BaseModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        # Common properties of NN models
        self.task_type = task_type
        self.num_classes = num_classes
        self.n_num_features = n_num_features
        self.n_cat_features = n_cat_features
        self.kg = kg
        self.metadata = metadata
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def compute_loss(self, pred, label):
        loss = 0.0
        ret = {}

        if self.task_type == 'regression':
            mse_loss = F.mse_loss(pred.flatten(), label)
            loss += mse_loss
            ret['mse'] = mse_loss
        elif self.task_type == 'binclass':
            xent_loss = F.binary_cross_entropy_with_logits(pred.flatten(), label.float())
            loss += xent_loss
            ret['xent'] = xent_loss
        elif self.task_type == 'multiclass':
            xent_loss = F.cross_entropy(pred, label)
            loss += xent_loss
            ret['xent'] = xent_loss

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


class BaseClassicalModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
