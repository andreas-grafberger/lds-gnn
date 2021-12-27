from typing import Optional

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data.utils import knn_graph, remove_edges, dense_adj_to_edge_index, shuffle_splits_, Transform, \
    largest_subgraph
from src.utils.graph import to_dense_adj
from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()


# noinspection Mypy
class KNNGraph(Transform):
    def __init__(self, loop: bool, k: int, metric: str = "cosine"):
        super().__init__()
        self.k = k
        self.loop = loop
        self.metric = metric

    def __call__(self, data: Data) -> Data:
        logger.info(f"Constructing knn-graph with k={self.k}, self-loop={self.loop}")
        copy = data.clone()
        copy.edge_index = knn_graph(copy.x, self.k, loop=self.loop, metric=self.metric)
        copy.dense_adj = to_dense_adj(copy.edge_index, num_max_nodes=data.num_nodes)
        return copy


# noinspection Mypy
class MakeUndirected(Transform):
    def __call__(self, data: Data) -> Data:
        logger.info(f"Making graph undirected (if not already)")
        copy = data.clone()
        copy.edge_index = to_undirected(data.edge_index, data.num_nodes)
        copy.dense_adj = to_dense_adj(copy.edge_index, num_max_nodes=data.num_nodes)
        return copy


# noinspection Mypy
class RemoveEdges(Transform):
    def __init__(self, remove_edges_percentage: float, seed: Optional[int] = None):
        super().__init__()
        assert 0.0 <= remove_edges_percentage <= 1.0
        self.remove_edges_percentage = remove_edges_percentage
        self.seed = seed

    def __call__(self, data: Data) -> Data:
        assert hasattr(data, "dense_adj")
        logger.info(f"Using {(1.0 - self.remove_edges_percentage) * 100}% of original edges")
        copy = data.clone()
        copy.dense_adj = remove_edges(copy.dense_adj, data.is_directed(), self.remove_edges_percentage, seed=self.seed)
        copy.edge_index = dense_adj_to_edge_index(copy.dense_adj)
        return copy


# noinspection Mypy
class ShuffleSplits(Transform):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

    def __call__(self, data: Data) -> Data:
        assert hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask")
        logger.info(f"Creating random splits")
        copy = data.clone()
        shuffle_splits_(copy, seed=self.seed)
        return copy


class CreateDenseAdjacencyMatrix(Transform):
    def __call__(self, data: Data) -> Data:
        copy = data.clone()
        copy.dense_adj = to_dense_adj(copy.edge_index, num_max_nodes=data.num_nodes)
        return copy


class LargestSubgraph(Transform):
    def __call__(self, data: Data) -> Data:
        logger.info(f"Using largest subgraph only (disconnected nodes are not removed!)")
        copy = data.clone()
        copy.edge_index = largest_subgraph(copy.edge_index, num_nodes=data.num_nodes)
        copy.dense_adj = to_dense_adj(copy.edge_index, num_max_nodes=data.num_nodes)
        return copy
