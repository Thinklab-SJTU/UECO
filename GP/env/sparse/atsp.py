import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from ml4co_kit import check_dim, to_tensor


def atsp_sparse_process(
    dists: np.ndarray, ref_tour: np.ndarray, sparse_factor: int
) -> Sequence[Tensor]:
    # check dimension
    check_dim(dists, 2)
    check_dim(ref_tour, 1)

    # nodes_num and edges_num
    nodes_num = dists.shape[0]
    edges_num = nodes_num * sparse_factor
    
    # KNN     
    idx_knn = np.argsort(dists, axis=1)[:, :sparse_factor]

    # edge_index
    edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
    edge_index_0 = edge_index_0.repeat(1, sparse_factor).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
    edge_index_src_tgt = torch.stack([edge_index_0, edge_index_1], dim=0)
    edge_index_tgt_src = torch.stack([edge_index_1, edge_index_0], dim=0)
    edge_index = torch.cat([edge_index_src_tgt, edge_index_tgt_src], dim=1) # (2, 2E)
    
    # edges_feature
    edges_feature_src_tgt = to_tensor(dists[edge_index_0, edge_index_1])
    edges_feature_tgt_src = to_tensor(dists[edge_index_1, edge_index_0])
    e = torch.cat([edges_feature_src_tgt, edges_feature_tgt_src])  # (2E,）
    # e = edges_feature_src_tgt
    
    # ground truth
    if ref_tour is not None:
        # tour_edges = np.zeros(nodes_num, dtype=np.int64)
        # tour_edges[ref_tour[:-1]] = ref_tour[1:]
        # tour_edges = torch.from_numpy(tour_edges)
        # tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        # ground_truth = torch.eq(edge_index_1, tour_edges).reshape(-1).long()

        successor = np.zeros(nodes_num, dtype=np.int64)
        successor[ref_tour[:-1]] = ref_tour[1:]
        successor = torch.from_numpy(successor)
        ground_truth = (edge_index[1] == successor[edge_index[0]]).long()
    else:
        ground_truth = None
    
    # other variables
    # e = torch.zeros(size=(edges_num,))

    return (
        None, # (V,): nodes feature, ATSP does not use it
        e.float(), # (2E,): k-nn edges feature, directional distance between nodes
        edge_index.long(), # (2, 2E): Index of edge enddists
        dists, # (V, V): Graph, full distance matrix
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
        edges_num # Number of edges
    )