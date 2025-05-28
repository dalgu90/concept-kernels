import os
import tempfile

import torch
import torch_geometric


def make_symmetric(edge_index, edge_weight):
    if edge_weight is not None:
        edge_index, edge_weight = torch_geometric.utils.coalesce(edge_index, edge_weight, reduce='mean')
        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight]) / 2.0
        edge_index, edge_weight = torch_geometric.utils.coalesce(edge_index, edge_weight)
        return edge_index, edge_weight
    else:
        edge_index = torch_geometric.utils.coalesce(edge_index)
        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)
        edge_index = torch_geometric.utils.coalesce(edge_index, reduce='mean')
        return edge_index, edge_weight

def get_normalized_laplacian(edge_index, edge_weight, num_nodes, remove_neg_edges=True):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1])
    if remove_neg_edges:
        pos_idxs = edge_weight > 0.0
        edge_index, edge_weight = edge_index[:, pos_idxs], edge_weight[pos_idxs]
    W = torch.zeros([num_nodes, num_nodes])
    W[edge_index[0], edge_index[1]] = edge_weight
    d = W.sum(dim=0).abs()
    d_sqrt = d ** -0.5
    d_sqrt[d == 0.0] = 0.0
    L = torch.eye(num_nodes) - d_sqrt.unsqueeze(1) * W * d_sqrt.unsqueeze(0)
    return L

def get_topk_eigen(A_sym, topk, largest=False):
    device = A_sym.device
    if torch.cuda.is_available():
        A_sym = A_sym.cuda()
    # eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
    U, S, Vh = torch.linalg.svd(A_sym)
    # S_large = S >= 1e-4
    # U, S, Vh = U[:, S_large], S[S_large], Vh[S_large, :]
    S_sign = (U.T / Vh).mean(dim=1).sign()
    eigenvalues = S * S_sign
    sorted_indices = torch.argsort(eigenvalues, descending=False)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = U[:, sorted_indices]
    eigenvectors *= torch.sign(eigenvectors[0, :])
    if not largest:
        eigenvalues = eigenvalues[:topk]
        eigenvectors = eigenvectors[:, :topk]
    else:
        eigenvalues = eigenvalues[-topk:].flip(dims=[0])
        eigenvectors = eigenvectors[:, -topk:].flip(dims=[1])
    eigenvalues, eigenvectors = eigenvalues.to(device), eigenvectors.to(device)
    return eigenvalues, eigenvectors

def get_global_eig_vecs(edge_weight_method, gsp_matrix_type, topk=200, data_dir='data/plato'):
    for prek in [250, 500, 1000]:
        if prek < topk: continue
        fname = f"eig_{edge_weight_method}_{gsp_matrix_type}_top{prek}.pt"
        fpath = os.path.join(data_dir, "kg", fname)
        if os.path.exists(fpath):
            eig_data = torch.load(fpath, weights_only=False)
            break
    eig_vecs = eig_data['eig_vecs'][:, :topk]
    return eig_vecs

def get_subgraph_edges(kg, node_idxs, edge_weight_method):
    edge_index, edge_attr = torch_geometric.utils.subgraph(
            node_idxs, kg.edge_index, kg.edge_attr, relabel_nodes=True)
    if edge_weight_method == 'uniform':
        edge_weight = torch.ones(edge_index.shape[1])
    elif edge_weight_method == 'cosine':
        V = kg.x[node_idxs]
        from_V, to_V = V[edge_index[0]], V[edge_index[1]]
        edge_weight = (from_V*to_V).sum(dim=1) / from_V.norm(dim=1) / to_V.norm(dim=1)
    elif edge_weight_method == 'score':
        V = kg.x[node_idxs]
        from_V, to_V = V[edge_index[0]], V[edge_index[1]]
        edge_weight = (from_V*kg.relation_embedding[edge_attr]*to_V).sum(dim=1)
        edge_weight = torch.log(1.0 + torch.clip(edge_weight, min=0.0))
    return edge_index, edge_weight

def get_local_eig_vecs(kg, metadata, gene_only, edge_weight_method, gsp_matrix_type, topk=200):
    X_mapping = metadata['X_mapping']
    if gene_only:
        X_mapping = X_mapping[:-metadata['num_drug_cols']]
    num_nodes = len(X_mapping)

    cache_fname = f"eig_{num_nodes}_{edge_weight_method}_{gsp_matrix_type}_500.pt"
    cache_fpath = os.path.join(tempfile.gettempdir(), cache_fname)
    if os.path.exists(cache_fpath):
        eig_vecs = torch.load(cache_fpath, weights_only=False)['eig_vecs']
        eig_vecs = eig_vecs[:, :topk]
        return eig_vecs

    edge_index, edge_weight = get_subgraph_edges(kg, X_mapping, edge_weight_method)
    edge_index, edge_weight = make_symmetric(edge_index, edge_weight)
    if gsp_matrix_type == 'adj':
        A = torch.zeros([num_nodes, num_nodes])
        A[edge_index[0], edge_index[1]] = edge_weight
    elif gsp_matrix_type == 'lap':
        A = get_normalized_laplacian(edge_index, edge_weight, num_nodes)
    eig_vals, eig_vecs = get_topk_eigen(A, 500, largest=(gsp_matrix_type=='adj'))
    torch.save({'eig_vals': eig_vals, 'eig_vecs': eig_vecs}, cache_fpath)
    eig_vecs = eig_vecs[:, :topk]
    return eig_vecs
