from itertools import product

import pytest
import torch

from src.utils.graph import DenseData
from tst.test_utils import cora  # Do not remove
from src.data.utils import knn_graph_dense
from src.data.transforms import *


# noinspection Mypy
@pytest.fixture()
def data(cora) -> DenseData:
    return CreateDenseAdjacencyMatrix()(cora[0])


@pytest.mark.parametrize("loop,k", product([True, False], [1, 5, 10]))
def test_knn_graph(data, loop, k):
    expected = knn_graph_dense(data.x, k=5, loop=False)
    computed = KNNGraph(loop=False, k=5)(data).dense_adj
    assert expected.equal(computed)


def test_make_undirected(data):
    expected = to_dense_adj(data.edge_index, num_max_nodes=data.num_nodes)
    computed = MakeUndirected()(data).dense_adj
    assert expected.equal(computed)


def test_remove_edges(data):
    computed = RemoveEdges(remove_edges_percentage=1.0)(data).dense_adj
    expected = torch.zeros((data.num_nodes, data.num_nodes))
    assert expected.equal(computed)


def test_remove_edges_returns_all(data):
    computed = RemoveEdges(remove_edges_percentage=0.0)(data).dense_adj
    expected = data.dense_adj
    assert expected.equal(computed)


def test_remove_edges_returns_half(data):
    computed = RemoveEdges(remove_edges_percentage=0.5)(data).dense_adj.sum()
    expected = data.dense_adj.sum() / 2
    assert torch.abs(computed - expected) <= 1


def test_remove_edges_throws_error(cora):
    with pytest.raises(AssertionError):
        computed = RemoveEdges(remove_edges_percentage=0.5)(cora[0]).dense_adj


def test_shuffle_splits(data):
    computed = ShuffleSplits(seed=123123123)(data)
    computed = torch.cat((computed.val_mask, computed.train_mask, computed.test_mask))
    shuffle_splits_(data, seed=123123123)
    expected = torch.cat((data.val_mask, data.train_mask, data.test_mask))
    assert expected.equal(computed)


def test_create_adjacency_matrix(cora):
    cora = cora[0]
    computed = CreateDenseAdjacencyMatrix()(cora).dense_adj
    expected = to_dense_adj(cora.edge_index, num_max_nodes=cora.num_nodes)
    assert expected.equal(computed)


def test_largest_subgraph(data):
    computed = LargestSubgraph()(data).dense_adj.sum().item()
    expected = 5069 * 2
    assert computed == expected
