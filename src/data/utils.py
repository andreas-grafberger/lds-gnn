from abc import ABC, abstractmethod
from os import path as osp
from pathlib import Path
from typing import Union, cast, List, Optional

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import scale
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

from src.utils.graph import is_square_matrix, DenseData, to_undirected
from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()

GRAPH_DATASETS = ["cora", "citeseer", "pubmed"]
UCI_DATASETS = ["digits", "wine", "breast_cancer"]


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data: Data) -> Data:
        pass


def load_uci_dataset(dataset: str):
    assert dataset in UCI_DATASETS

    data_object: DenseData = DenseData()

    if dataset == "digits":
        data = sklearn_datasets.load_digits()
        data_object.x = torch.as_tensor(data.get("data")).float().view(-1, 8 * 8)  # Flatten images
        data_object.y = torch.as_tensor(data.get("target")).long()
        train_mask_size, val_mask_size, test_mask_size = (50, 100, -1)
    elif dataset == "wine":
        data = sklearn_datasets.load_wine()
        data_object.x = torch.as_tensor(scale(data.get("data"))).float()
        data_object.y = torch.as_tensor(data.get("target")).long()
        train_mask_size, val_mask_size, test_mask_size = (10, 20, -1)
    elif dataset == "breast_cancer":
        data = sklearn_datasets.load_breast_cancer()
        data_object.x = torch.as_tensor(scale(data.get("data"))).float()
        data_object.y = torch.as_tensor(data.get("target")).long()
        train_mask_size, val_mask_size, test_mask_size = (10, 20, -1)
    else:
        raise NotImplementedError()

    data_object.num_nodes = data_object.x.size(0)
    data_object.edge_index = torch.eye(data_object.num_nodes).nonzero().t()
    if test_mask_size == -1:
        test_mask_size = data_object.num_nodes - (train_mask_size + val_mask_size)

    empty_mask = torch.zeros(data_object.num_nodes)
    data_object.train_mask = empty_mask.index_fill(dim=0, index=torch.arange(0, train_mask_size), value=1.0).bool()
    data_object.val_mask = empty_mask.index_fill(dim=0, index=torch.arange(train_mask_size,
                                                                           train_mask_size + val_mask_size),
                                                 value=1.0).bool()

    data_object.test_mask = empty_mask.index_fill(dim=0,
                                                  index=torch.arange(train_mask_size + val_mask_size,
                                                                     train_mask_size + val_mask_size + test_mask_size),
                                                  value=1.0).bool()
    return data_object


def load_planetoid_dataset(dataset: str, path: Union[Path, str] = None) -> Planetoid:
    logger.info(f"Loading dataset {dataset}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', dataset) if path is None else path
    return Planetoid(path, dataset, transform=None)


def load_planetoid_dataset_with_transforms(dataset: str,
                                           transform: Optional[Transform] = None,
                                           path: Union[Path, str] = None,
                                           ) -> Planetoid:
    logger.info(f"Loading dataset {dataset} and uses transforms {transform}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', dataset) if path is None else path
    return Planetoid(path, dataset, transform=transform)


def unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    assert (edge_index.size(0) == 0) or (len(edge_index.size()) == 2 and edge_index.size(0) == 2)
    if edge_index.size(0) == 0 or edge_index.size(1) == 0:
        return edge_index
    else:
        return torch.unique(edge_index, dim=1)  # type: ignore


def filter_edges(edge_index: torch.Tensor, nodes_to_keep: List[int]) -> torch.Tensor:
    edges = edge_index.t().numpy()
    allowed_edges = [(a, b) for (a, b) in edges if a in nodes_to_keep or b in nodes_to_keep]
    # noinspection PyArgumentList,Mypy
    edges = torch.Tensor(allowed_edges).t().to(edge_index.device).long()
    return unique_edges(edges)


def largest_subgraph(edge_index: torch.Tensor,
                     n_components: int = 1,
                     num_nodes: int = None,
                     ) -> torch.Tensor:
    """
    Implementation inspired by "https://github.com/shchur/gnn-benchmark/blob/
    1e72912a0810cdf27ae54fd589a3b43358a2b161/gnnbench/data/preprocess.py#L61"
    """
    assert len(edge_index.size()) == 2 and edge_index.size(0) == 2
    logger.info(f"Only using largest subgraph")
    sparse_matrix = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes)
    _, indices = connected_components(sparse_matrix,
                                      directed=True)  # Directed should not matter when using pytorch-geometric graph
    sizes = np.bincount(indices)
    to_keep = np.argsort(sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(indices) if component in to_keep
    ]
    return filter_edges(edge_index, nodes_to_keep)


def indices_to_mask(indices: torch.LongTensor, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=indices.device)
    mask[indices] = 1
    return mask


def dense_adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, adj.nonzero().t())


def shuffle_splits_(data: DenseData, seed=None) -> None:
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    train_size, val_size, test_size = train_mask.sum(), val_mask.sum(), test_mask.sum()

    splitter = StratifiedShuffleSplit(n_splits=1,
                                      test_size=test_size,
                                      train_size=val_size + train_size,
                                      random_state=seed)
    train_val_indices, test_indices = next(splitter.split(data.x, data.y))

    train_val_splitter = StratifiedShuffleSplit(n_splits=1,
                                                test_size=val_size,
                                                train_size=train_size,
                                                random_state=seed)
    train_indices, val_indices = next(train_val_splitter.split(data.x[train_val_indices], data.y[train_val_indices]))
    train_indices, val_indices = train_val_indices[train_indices], train_val_indices[val_indices]

    train_indices = torch.as_tensor(train_indices, device=data.x.device)
    val_indices = torch.as_tensor(val_indices, device=data.x.device)
    test_indices = torch.as_tensor(test_indices, device=data.x.device)

    train_mask = indices_to_mask(train_indices, train_mask.size(0))
    val_mask = indices_to_mask(val_indices, val_mask.size(0))
    test_mask = indices_to_mask(test_indices, test_mask.size(0))

    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask


def knn_graph_dense(x: torch.Tensor,
                    k: int,
                    loop: bool = True,
                    metric: str = "cosine") -> torch.Tensor:
    from sklearn.neighbors import kneighbors_graph
    graph = kneighbors_graph(x.numpy(),
                             n_neighbors=k,
                             mode="connectivity",
                             metric=metric,
                             include_self=loop)
    return torch.FloatTensor(graph.toarray())


def knn_graph(x: torch.Tensor,
              k: int,
              loop: bool = True,
              metric: str = "cosine") -> torch.Tensor:
    adj = knn_graph_dense(x=x, k=k, loop=loop, metric=metric)
    return dense_adj_to_edge_index(adj)


def remove_edges(dense_adj: torch.Tensor,
                 is_directed: bool,
                 remove_edges_percentage: float,
                 seed: int = None) -> torch.Tensor:
    logger.info(f"Removing {remove_edges_percentage} percent of edges in the graph.")
    if is_directed:
        return remove_edges_from_directed_graph(dense_adj, remove_edges_percentage, seed=seed)
    else:
        return remove_edges_from_undirected_graph(dense_adj, remove_edges_percentage, seed=seed)


def remove_edges_from_directed_graph(adj: torch.Tensor, remove_edges_percentage: float,
                                     seed: int = None) -> torch.Tensor:
    assert 0.0 <= remove_edges_percentage <= 1.0
    assert is_square_matrix(adj)

    nonzero_indices = adj.nonzero()
    num_edges = nonzero_indices.size(0)
    num_edges_to_keep = int(num_edges * (1.0 - remove_edges_percentage))

    with PytorchSeedOverwrite(seed):
        perm = torch.randperm(nonzero_indices.size(0))

    idx = perm[:num_edges_to_keep]
    edges_to_retain = nonzero_indices.t()[:, idx]

    new_adj = torch.zeros_like(adj, device=adj.device)
    new_adj[edges_to_retain[0], edges_to_retain[1]] = adj[edges_to_retain[0], edges_to_retain[1]]
    return new_adj


def remove_edges_from_undirected_graph(adj: torch.Tensor,
                                       remove_edges_percentage: float,
                                       seed: int = None) -> torch.Tensor:
    assert 0.0 <= remove_edges_percentage <= 1.0
    assert is_square_matrix(adj)
    assert adj.t().equal(adj)

    adj_clone = adj.clone().triu()
    adj_removed = remove_edges_from_directed_graph(adj_clone, remove_edges_percentage, seed=seed)
    adj_sym = to_undirected(adj_removed, from_triu_only=True)
    return adj_sym


class PytorchSeedOverwrite:

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __enter__(self):
        self.random_state = torch.random.get_rng_state()
        if self.seed is not None:
            torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.random.set_rng_state(self.random_state)
