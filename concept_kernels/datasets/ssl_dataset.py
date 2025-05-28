import numpy as np
import torch

from concept_kernels.datasets.base_dataset import BaseDataset


class SSLTransitionDataset():
    def __init__(self, base_dataset: BaseDataset, num_transition=1, scale_jitter=0.0):
        self.base_dataset = base_dataset
        self.num_transition = num_transition
        self.scale_jitter = scale_jitter
        self.T_num, self.T_cat = None, None
        n_num_features = self.metadata['n_num_features']
        n_cat_features = self.metadata['n_cat_features']
        if n_num_features:
            if 'col_sim_mat' in self.metadata:
                W_num = self.metadata['col_sim_mat'][:n_num_features, :n_num_features]
            else:
                V_num = self.kg.x[:n_num_features]
                W_num = V_num @ V_num.T
            self.T_num, self.pi_num = self.get_transition(W_num)
        if n_cat_features:
            if 'col_sim_mat' in self.metadata:
                W_cat = self.metadata['col_sim_mat'][-n_cat_features:, -n_cat_features:]
            else:
                V_cat = self.kg.x[-n_cat_features:]
                W_cat = V_cat @ V_cat.T
            self.T_cat, self.pi_cat = self.get_transition(W_cat)
            # categories: [3, 4, 2]
            # -> cat_to_col: [0,0,0,1,1,1,1,2,2], cat_offset: [0, 3, 7, 9]
            self.cat_to_col, self.cat_offset = [], [0]
            for col_idx, n_cat in enumerate(self.metadata['categories']):
                self.cat_to_col += [col_idx] * n_cat
                self.cat_offset.append(self.cat_offset[-1] + n_cat)
        if 'col_sim_mat' in self.metadata:
            W = self.metadata['col_sim_mat']
        else:
            W = self.kg.x @ self.kg.x.T
        self.T, self.pi = self.get_transition(W)
        if n_num_features < 2:
            self.pi[:n_num_features] = 0.0

    # Delegated access to the attr of base_dataset
    _delegated_attrs = {'X_num', 'X_cat', 'y', 'split', 'is_preprocessed', 'kg', 'metadata'}

    def __getattr__(self, name):
        if name in self._delegated_attrs:
            return getattr(self.base_dataset, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self._delegated_attrs:
            setattr(self.base_dataset, name, value)
        else:
            super().__setattr__(name, value)

    def get_transition(self, W):
        W = torch.clamp(W, min=0.0)
        D = W.sum(dim=0, keepdims=True)
        T =  W * (D ** -1)  # Left stochastic matrix (col sum is 1.0)
        eig_vals, eig_vecs = torch.linalg.eig(T)
        eig_indices = torch.argsort(eig_vals.abs(), descending=True)
        eig_vals, eig_vecs = eig_vals[eig_indices], eig_vecs[:, eig_indices]
        assert torch.allclose(eig_vals[0].real, torch.tensor(1.0)), "First eigen value != 1.0"
        assert torch.allclose(eig_vals[0].imag, torch.tensor(0.0)), "First eigen value != 1.0"
        pi = eig_vecs[:, 0].real
        pi = pi.abs() / pi.norm(p=1)
        return T, pi

    def __len__(self):
        return len(self.base_dataset)

    def random_swap(self, X_num, X_cat):
        n_num_features = self.metadata['n_num_features']
        n_cat_features = self.metadata['n_cat_features']
        perm_num = None
        if n_num_features:
            perm_num = torch.arange(n_num_features)

        dist_i = []
        if X_num is not None:
            dist_i.append(self.pi[:n_num_features])
        if X_cat is not None:
            X_cat = X_cat.bool()
            dist_i.append(self.pi[-n_cat_features:][X_cat])
            X_cat = torch.where(X_cat)[0]
        dist_i = torch.cat(dist_i)

        for _ in range(self.num_transition):
            i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            if i < n_num_features:  # numerical feature -> choice and swap
                dist_j = self.T_num[:,i].clone()
                dist_j[i] = 0.0
                j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            else:
                i -= n_num_features
                dist_cat = self.T_cat[:, X_cat[i]].clone()
                dist_cat[X_cat[i]] = 0.0
                new_cat = torch.multinomial(dist_cat, num_samples=1, replacement=True).item()
                X_cat[i] = new_cat
                dist_i[i+n_num_features] = self.pi[new_cat]
        return X_num, X_cat, perm_num

    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        X_num, X_cat = example['X_num'], example['X_cat']
        X_num1, X_cat1, X_num2, X_cat2 = None, None, None, None
        if X_num is not None:
            X_num1, X_num2 = X_num.clone(), X_num.clone()
        if X_cat is not None:
            X_cat1, X_cat2 = X_cat.clone(), X_cat.clone()

        X_num1, X_cat1, perm_num1 = self.random_swap(X_num1, X_cat1)
        X_num2, X_cat2, perm_num2 = self.random_swap(X_num2, X_cat2)

        ret = {
            'X_num1': X_num1,
            'X_cat1': X_cat1,
            'perm_num1': perm_num1,
            'X_num2': X_num2,
            'X_cat2': X_cat2,
            'perm_num2': perm_num2,
            'label': example['label']
        }
        return ret

    def collate_fn(self, examples):
        # X_cat and X_num can be None, but not both at the same time
        X_num1, X_cat1, perm_num1, X_num2, X_cat2, perm_num2 = None, None, None, None, None, None
        if examples[0]['X_num1'] is not None:
            X_num1 = torch.stack([e['X_num1'] for e in examples])
            X_num2 = torch.stack([e['X_num2'] for e in examples])
            perm_num1 = torch.stack([e['perm_num1'] for e in examples])
            perm_num2 = torch.stack([e['perm_num2'] for e in examples])
        if examples[0]['X_cat1'] is not None:
            X_cat1 = torch.stack([e['X_cat1'] for e in examples])
            X_cat2 = torch.stack([e['X_cat2'] for e in examples])
        ret = {
            'X_num1': X_num1,
            'X_cat1': X_cat1,
            'perm_num1': perm_num1,
            'X_num2': X_num2,
            'X_cat2': X_cat2,
            'perm_num2': perm_num2,
            'label': torch.stack([e['label'] for e in examples])
        }
        return ret

# class SSLTransitionDataset2(SSLTransitionDataset):
    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)
        # dist_i = self.pi.clone()
        # if X_cat is not None:
            # dist_i[-n_cat_features:] = dist_i[-n_cat_features:] * X_cat

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # ii = i - n_num_features
                # ii_cat = self.cat_to_col[ii]
                # dist_jj = self.T_cat[:, ii].clone()
                # dist_jj[:self.cat_offset[ii_cat]] = 0.0
                # dist_jj[ii] = 0.0
                # dist_jj[self.cat_offset[ii_cat+1]:] = 0.0
                # jj = torch.multinomial(dist_jj, num_samples=1, replacement=True).item()
                # X_cat.scatter_(0, torch.tensor([jj, ii]), X_cat[[ii, jj]].clone())
                # ii, jj = ii + n_num_features, jj + n_num_features
        # return X_num, X_cat, perm_num

    # def __getitem__(self, idx):
        # example = self.base_dataset[idx]
        # X_num, X_cat = example['X_num'], example['X_cat']
        # X_num1, X_cat1, X_num2, X_cat2 = None, None, None, None
        # if X_num is not None:
            # X_num1, X_num2 = X_num.clone(), X_num.clone()
        # if X_cat is not None:
            # X_cat1, X_cat2 = X_cat.clone(), X_cat.clone()

        # X_num1, X_cat1, perm_num1 = self.random_swap(X_num1, X_cat1)
        # X_num2, X_cat2, perm_num2 = self.random_swap(X_num2, X_cat2)

        # ret = {
            # 'X_num1': X_num1,
            # 'X_cat1': X_cat1,
            # 'perm_num1': perm_num1,
            # 'X_num2': X_num2,
            # 'X_cat2': X_cat2,
            # 'perm_num2': perm_num2,
            # 'label': example['label']
        # }
        # return ret

    # def collate_fn(self, examples):
        # # X_cat and X_num can be None, but not both at the same time
        # X_num1, X_cat1, perm_num1, X_num2, X_cat2, perm_num2 = None, None, None, None, None, None
        # if examples[0]['X_num1'] is not None:
            # X_num1 = torch.stack([e['X_num1'] for e in examples])
            # X_num2 = torch.stack([e['X_num2'] for e in examples])
            # perm_num1 = torch.stack([e['perm_num1'] for e in examples])
            # perm_num2 = torch.stack([e['perm_num2'] for e in examples])
        # if examples[0]['X_cat1'] is not None:
            # X_cat1 = torch.stack([e['X_cat2'] for e in examples])
            # X_cat2 = torch.stack([e['X_cat2'] for e in examples])
        # ret = {
            # 'X_num1': X_num1,
            # 'X_cat1': X_cat1,
            # 'perm_num1': perm_num1,
            # 'X_num2': X_num2,
            # 'X_cat2': X_cat2,
            # 'perm_num2': perm_num2,
            # 'label': torch.stack([e['label'] for e in examples])
        # }
        # return ret

# class SSLTransitionDataset3(SSLTransitionDataset2):
    # def __init__(self, base_dataset: BaseDataset, num_transition=1, scale_jitter=0.0):
        # super().__init__(base_dataset, num_transition, scale_jitter)
        # n_cat_features = self.metadata['n_cat_features']
        # if 'categories' in self.metadata and len(self.metadata['categories']) == 1:
            # self.pi[-n_cat_features:] = 0.0

    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # perm_num = None
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)
        # dist_i = []
        # if X_num is not None:
            # dist_i.append(self.pi[:n_num_features])
        # if X_cat is not None:
            # X_cat = X_cat.bool()
            # dist_i.append(self.pi[-n_cat_features:][X_cat])
            # T_cat = self.T_cat[X_cat][:, X_cat]
            # X_cat = torch.where(X_cat)[0]
        # dist_i = torch.cat(dist_i)

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # i -= n_num_features
                # dist_j = T_cat[:, i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_cat.scatter_(0, torch.tensor([j, i]), X_cat[[i, j]].clone())
                # i += n_num_features
                # j += n_num_features
                # dist_i.scatter_(0, torch.tensor([j, i]), dist_i[[i, j]].clone())
        # return X_num, X_cat, perm_num


# class SSLTransitionDataset4(SSLTransitionDataset2):
    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # perm_num = None
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)

        # dist_i = []
        # if X_num is not None:
            # dist_i.append(self.pi[:n_num_features])
        # if X_cat is not None:
            # X_cat = X_cat.bool()
            # dist_i.append(self.pi[-n_cat_features:][X_cat])
            # X_cat = torch.where(X_cat)[0]
        # dist_i = torch.cat(dist_i)

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # i -= n_num_features
                # dist_cat = self.T_cat[:, X_cat[i]].clone()
                # dist_cat[X_cat[i]] = 0.0
                # new_cat = torch.multinomial(dist_cat, num_samples=1, replacement=True).item()
                # X_cat[i] = new_cat
                # dist_i[i+n_num_features] = self.pi[new_cat]
        # return X_num, X_cat, perm_num


# class SSLTransitionDataset5(SSLTransitionDataset2):
    # def __init__(self, base_dataset: BaseDataset, num_transition=1, scale_jitter=0.0):
        # self.base_dataset = base_dataset
        # self.num_transition = num_transition
        # self.scale_jitter = scale_jitter
        # self.T_num, self.T_cat = None, None
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # categories = self.metadata.get('categories', None)
        # if n_num_features:
            # V_num = self.kg.x[:n_num_features]
            # self.T_num, self.pi_num = self.get_transition(V_num)
        # if n_cat_features:
            # V_cat = self.kg.x[-n_cat_features:]
            # self.T_cat, self.pi_cat = self.get_transition(V_cat, categories, 0)
            # # categories: [3, 4, 2]
            # # -> cat_to_col: [0,0,0,1,1,1,1,2,2], cat_offset: [0, 3, 7, 9]
            # self.cat_to_col, self.cat_offset = [], [0]
            # for col_idx, n_cat in enumerate(categories):
                # self.cat_to_col += [col_idx] * n_cat
                # self.cat_offset.append(self.cat_offset[-1] + n_cat)
        # V = self.kg.x
        # self.T, self.pi = self.get_transition(V, categories, n_num_features)
        # if n_num_features < 2:
            # self.pi[:n_num_features] = 0.0
        # if categories is not None and len(categories) == 1:
            # self.pi[-n_cat_features:] = 0.0

    # def get_transition(self, embed, categories=None, cat_offset=0, temp=0.2):
        # W = embed @ embed.T
        # if categories:
            # co = cat_offset
            # for nc in categories:
                # e = embed[co:co+nc]
                # e_mean = e.mean(dim=0, keepdims=True)
                # e = torch.nn.functional.normalize(e - e_mean, p=2, dim=1)
                # W[np.ix_(range(co,co+nc), range(co,co+nc))] = e @ e.T
                # co += nc
        # W = (W / temp).exp()
        # D = W.sum(dim=0, keepdims=True)
        # T =  W * (D ** -1)  # Left stochastic matrix (col sum is 1.0)
        # eig_vals, eig_vecs = torch.linalg.eig(T)
        # eig_indices = torch.argsort(eig_vals.abs(), descending=True)
        # eig_vals, eig_vecs = eig_vals[eig_indices], eig_vecs[:, eig_indices]
        # assert torch.allclose(eig_vals[0].real, torch.tensor(1.0)), "First eigen value != 1.0"
        # assert torch.allclose(eig_vals[0].imag, torch.tensor(0.0)), "First eigen value != 1.0"
        # pi = eig_vecs[:, 0].real
        # pi = pi.abs() / pi.norm(p=1)
        # return T, pi

    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # perm_num = None
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)

        # dist_i = []
        # if X_num is not None:
            # dist_i.append(self.pi[:n_num_features])
        # if X_cat is not None:
            # X_cat = X_cat.bool()
            # dist_i.append(self.pi[-n_cat_features:][X_cat])
            # X_cat = torch.where(X_cat)[0]
        # dist_i = torch.cat(dist_i)

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # i -= n_num_features
                # dist_cat = self.T_cat[:, X_cat[i]].clone()
                # dist_cat[X_cat[i]] = 0.0
                # new_cat = torch.multinomial(dist_cat, num_samples=1, replacement=True).item()
                # X_cat[i] = new_cat
                # dist_i[i+n_num_features] = self.pi[new_cat]
        # return X_num, X_cat, perm_num


# class SSLTransitionDataset6(SSLTransitionDataset4):
    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # perm_num = None
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)

        # dist_i = []
        # if X_num is not None:
            # dist_i.append(self.pi[:n_num_features])
        # if X_cat is not None:
            # X_cat = X_cat.bool()
            # dist_i.append(self.pi[-n_cat_features:][X_cat])
            # X_cat = torch.where(X_cat)[0]
        # dist_i = torch.cat(dist_i)

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # i -= n_num_features
                # dist_cat = self.T_cat[:, X_cat[i]].clone()
                # dist_cat[X_cat[i]] = 0.0
                # new_cat = torch.multinomial(dist_cat, num_samples=1, replacement=True).item()
                # X_cat[i] = new_cat
                # dist_i[i+n_num_features] = self.pi[new_cat]
        # if X_num is not None:
            # X_num += torch.randn_like(X_num) * self.scale_jitter
        # return X_num, X_cat, perm_num


# class SSLTransitionDataset7(SSLTransitionDataset2):
    # def __init__(self, base_dataset: BaseDataset, num_transition=1, scale_jitter=0.0):
        # self.base_dataset = base_dataset
        # self.num_transition = num_transition
        # self.scale_jitter = scale_jitter
        # self.T_num, self.T_cat = None, None
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # categories = self.metadata.get('categories', None)
        # category_ranks = self.metadata.get('category_ranks', None)
        # if n_num_features:
            # V_num = self.kg.x[:n_num_features]
            # self.T_num, self.pi_num = self.get_transition(V_num)
        # if n_cat_features:
            # V_cat = self.kg.x[-n_cat_features:]
            # self.T_cat, self.pi_cat = self.get_transition(V_cat, categories, category_ranks, 0)
            # # categories: [3, 4, 2]
            # # -> cat_to_col: [0,0,0,1,1,1,1,2,2], cat_offset: [0, 3, 7, 9]
            # self.cat_to_col, self.cat_offset = [], [0]
            # for col_idx, n_cat in enumerate(categories):
                # self.cat_to_col += [col_idx] * n_cat
                # self.cat_offset.append(self.cat_offset[-1] + n_cat)
        # V = self.kg.x
        # self.T, self.pi = self.get_transition(V, categories, category_ranks, n_num_features)
        # if n_num_features < 2:
            # self.pi[:n_num_features] = 0.0
        # if categories is not None and len(categories) == 1:
            # self.pi[-n_cat_features:] = 0.0

    # def get_transition(self, embed, categories=None, category_ranks=None,
                       # cat_offset=0, alpha=0.9, sigma=0.25):
        # W = embed @ embed.T
        # if categories:
            # co = cat_offset
            # for nc, cr in zip(categories, category_ranks):
                # cat_range_ix = np.ix_(range(co,co+nc), range(co,co+nc))
                # cr = torch.tensor(cr, dtype=torch.float)
                # cr -= cr.min()
                # cr /= cr.max()
                # rank_score = torch.exp(-((cr[:, None]-cr[None, :])**2)/(2*sigma**2))
                # # W[cat_range_ix] = W[cat_range_ix]*(1-alpha) + rank_score*alpha
                # W[cat_range_ix] = W[cat_range_ix] + rank_score*alpha/(1-alpha)
                # co += nc
        # W = torch.clamp(W, min=0)
        # D = W.sum(dim=0, keepdims=True)
        # T =  W * (D ** -1)  # Left stochastic matrix (col sum is 1.0)
        # eig_vals, eig_vecs = torch.linalg.eig(T)
        # eig_indices = torch.argsort(eig_vals.abs(), descending=True)
        # eig_vals, eig_vecs = eig_vals[eig_indices], eig_vecs[:, eig_indices]
        # assert torch.allclose(eig_vals[0].real, torch.tensor(1.0)), "First eigen value != 1.0"
        # assert torch.allclose(eig_vals[0].imag, torch.tensor(0.0)), "First eigen value != 1.0"
        # pi = eig_vecs[:, 0].real
        # pi = pi.abs() / pi.norm(p=1)
        # return T, pi

    # def random_swap(self, X_num, X_cat):
        # n_num_features = self.metadata['n_num_features']
        # n_cat_features = self.metadata['n_cat_features']
        # perm_num = None
        # if n_num_features:
            # perm_num = torch.arange(n_num_features)

        # dist_i = []
        # if X_num is not None:
            # dist_i.append(self.pi[:n_num_features])
        # if X_cat is not None:
            # X_cat = X_cat.bool()
            # dist_i.append(self.pi[-n_cat_features:][X_cat])
            # X_cat = torch.where(X_cat)[0]
        # dist_i = torch.cat(dist_i)

        # for _ in range(self.num_transition):
            # i = torch.multinomial(dist_i, num_samples=1, replacement=True).item()
            # if i < n_num_features:  # numerical feature -> choice and swap
                # dist_j = self.T_num[:,i].clone()
                # dist_j[i] = 0.0
                # j = torch.multinomial(dist_j, num_samples=1, replacement=True).item()
                # X_num.scatter_(0, torch.tensor([j, i]), X_num[[i, j]].clone())
                # perm_num.scatter_(0, torch.tensor([j, i]), perm_num[[i, j]].clone())
            # else:
                # i -= n_num_features
                # dist_cat = self.T_cat[:, X_cat[i]].clone()
                # dist_cat[X_cat[i]] = 0.0
                # new_cat = torch.multinomial(dist_cat, num_samples=1, replacement=True).item()
                # X_cat[i] = new_cat
                # dist_i[i+n_num_features] = self.pi[new_cat]
        # return X_num, X_cat, perm_num
