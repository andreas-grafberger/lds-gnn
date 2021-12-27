from math import sqrt
from typing import Union, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_add

from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()


class DenseData(Data):

    def __init__(self, **kwargs):
        super(DenseData).__init__(**kwargs)
        self.dense_adj: torch.Tensor = None
        self.train_mask: torch.Tensor = None
        self.val_mask: torch.Tensor = None
        self.test_mask: torch.Tensor = None
        self.num_classes: int = -1
        self.name: str = ""


def to_undirected(adj: torch.Tensor,
                  from_triu_only: bool = False) -> torch.Tensor:
    assert is_square_matrix(adj)

    if not from_triu_only:
        result = torch.max(adj, adj.t())
        #result = (adj + adj.t()).clamp(0.0, 1.0)  # TODO Does not work for weighted graphs
    else:
        triu = adj.triu(1)
        samples_without_self_loops = triu + triu.t()
        result = samples_without_self_loops + torch.diag(adj.diag())
    return result


def get_triu_values(adj: torch.Tensor) -> torch.Tensor:
    assert adj.size(0) == adj.size(1)
    n_nodes = adj.size(0)
    indices = torch.triu_indices(n_nodes, n_nodes, device=adj.device)
    return adj[indices[0], indices[1]]


def split_mask(mask: torch.Tensor,
               ratio: float = 0.5,
               shuffle: bool = True,
               device: Union[str, torch.device] = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the specified mask into two parts. Can be used to extract a (random) subset of the validation set only used
    to optimize some other loss/ parameters.
    :param mask: Tensor with shape [N_NODES] with dtype torch.bool
    :param ratio: Proportion of validation samples in first subset (range: 0.0-1.0)
    :param shuffle: whether splits contain random samples from validation set
    :param device: device on which new tensors are stored
    :return: Tuple with both new masks
    """
    nonzero_indices = mask.nonzero()

    if shuffle:
        shuffled_indices = np.arange(nonzero_indices.size(0))
        np.random.shuffle(shuffled_indices)
        nonzero_indices = nonzero_indices[shuffled_indices]

    split_index = int(nonzero_indices.size(0) * ratio)
    first_part_indices = nonzero_indices[:split_index]
    second_part_indices = nonzero_indices[split_index:]

    first_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
    first_mask[first_part_indices] = 1
    second_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
    second_mask[second_part_indices] = 1
    return first_mask, second_mask


# COPIED FROM PYTORCH_GEOMETRIC LIBRARY AND MODIFIED
def to_dense_adj(edge_index, batch=None, edge_attr=None, num_max_nodes=None) -> torch.Tensor:
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = num_nodes.max().item() if not num_max_nodes else num_max_nodes

    size = [batch_size, max_num_nodes, max_num_nodes]
    size = size if edge_attr is None else size + list(edge_attr.size())[1:]
    dtype = torch.float if edge_attr is None else edge_attr.dtype
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if edge_attr is None:
        adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        adj[edge_index_0, edge_index_1, edge_index_2] = edge_attr

    return adj.squeeze()


def is_square_matrix(tensor: torch.Tensor) -> bool:
    return len(tensor.size()) == 2 and tensor.size(0) == tensor.size(1)


def add_self_loops(adj: torch.Tensor):
    """
    Adds self loop to graph by setting diagonal of adjacency matrix to 1.
    Preserves gradient flow
    :param adj: Square matrix
    :return: Cloned tensor with diagonals set to 1
    """
    assert is_square_matrix(adj)
    adj_clone = adj.clone()
    adj_clone.fill_diagonal_(1.0)
    return adj_clone


def normalize_adjacency_matrix(dense_adj: torch.Tensor) -> torch.Tensor:
    """
    Normalizes adjacency matrix as proposed in original GCN paper.
    :param dense_adj: Dense adjacency matrix
    :return:
    """
    assert is_square_matrix(dense_adj)

    # Add self-loops
    dense_adj_with_self_loops = add_self_loops(dense_adj)

    # Normalization
    degree_matrix = dense_adj_with_self_loops.sum(dim=1)
    inv_sqrt_degree_matrix = 1.0 / degree_matrix.sqrt()
    inv_sqrt_degree_matrix = torch.diag(inv_sqrt_degree_matrix).to(dense_adj.device)

    normalized_dense_adj = inv_sqrt_degree_matrix @ dense_adj_with_self_loops @ inv_sqrt_degree_matrix
    return normalized_dense_adj


def cosine_similarity(a: Tensor, b: Optional[Tensor] = None, eps: Union[float, Tensor] = 1e-8) -> Tensor:
    a_norm = a.norm(p=2, dim=1, keepdim=True)
    if b is None:
        b = a
        b_norm = a_norm
    else:
        b_norm = b.norm(p=2, dim=1, keepdim=True)
    return (torch.mm(a, b.t()) / (a_norm * b_norm.t()).clamp(min=eps)).clamp_max(1.0)  # type: ignore


def triu_values_to_symmetric_matrix(triu_values: Tensor) -> Tensor:
    """
    Given the flattened tensor of values in the triangular matrix, constructs a symmetric adjacency matrix
    :param triu_values: Flattened tensor of shape (TRIANG_N,)
    :return: Symmetric adjacency matrix of shape (N, N)
    """
    assert len(triu_values.size()) == 1
    n_nodes = num_nodes_from_triu_shape(triu_values.size(0))
    indices = torch.triu_indices(n_nodes, n_nodes, device=triu_values.device)
    adj = torch.zeros((n_nodes, n_nodes), device=triu_values.device)
    adj[indices[0], indices[1]] = triu_values

    adj = to_undirected(adj, from_triu_only=True)

    adj = adj.clamp(0.0, 1.0)
    return adj


def num_nodes_from_triu_shape(n_triu_values: int) -> int:
    """
    A (slightly hacky) way to calculate the number of nodes in the original graph given the number of nodes in the
    upper/ lower triangular matrix
    :param n_triu_values: Number of nodes in triangular matrix
    :return: Number of nodes in graph
    """
    n_nodes = int(0.5 * sqrt((8 * n_triu_values + 1) - 1))
    return n_nodes


def dirichlet_energy(adj: Tensor, features: Tensor) -> Tensor:
    degree_matrix = adj.sum(dim=1).diag()
    laplacian = degree_matrix - adj
    smoothness_matrix = features.t() @ laplacian @ features
    loss = smoothness_matrix.trace() / adj.numel()
    # noinspection Mypy
    return loss


def disconnection_loss(adj: Tensor) -> Tensor:
    # noinspection Mypy
    return -adj.size(0) * (adj.sum(dim=1) + 10e-8).log().sum()


def sparsity_loss(adj: Tensor) -> Tensor:
    frob_norm_squared = (adj * adj).sum()
    return frob_norm_squared / adj.numel()


def graph_regularization(graph: Tensor,
                         features: Tensor,
                         smoothness_factor: float,
                         disconnection_factor: float,
                         sparsity_factor: float,
                         log: bool = True
                         ) -> Tensor:
    _smoothness_loss = dirichlet_energy(graph, features=features)
    _disconnection_loss = disconnection_loss(graph)
    _sparsity_loss = sparsity_loss(graph)

    if log:
        logger.info(
            f"Regularization-losses: Smoothness={_smoothness_loss.item()}, "
            f"Disconnection={_disconnection_loss.item()}, "
            f"Sparsity={_sparsity_loss}"
        )

    return smoothness_factor * _smoothness_loss + \
           disconnection_factor * _disconnection_loss + \
           sparsity_factor * _sparsity_loss
