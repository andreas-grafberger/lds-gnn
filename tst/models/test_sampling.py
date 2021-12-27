from itertools import product
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor
from torch.distributions import Bernoulli

from src.models.sampling import sample_graph, straight_through_estimator, SPARSIFICATION, sparsify

NUM_NODES = 20
EMB_DIM = 32


@pytest.fixture
def directed_graph_params() -> Tensor:
    return torch.ones((NUM_NODES, NUM_NODES)).triu(diagonal=1).requires_grad_(True)


@pytest.fixture
def embeddings() -> Tensor:
    emb = torch.arange(0.0, 1.0, step=1.0 / NUM_NODES)[:, None]
    emb = emb.expand(-1, EMB_DIM)
    return emb


@pytest.mark.parametrize(argnames=["undirected",
                                   "sparsification",
                                   "dense",
                                   "k"],
                         argvalues=product([True, False],
                                           [SPARSIFICATION.NONE, SPARSIFICATION.KNN],
                                           [True, False],
                                           [10]))
def test_sample_graph_always_differentiable(directed_graph_params: Tensor,
                                            embeddings: Tensor,
                                            undirected: bool,
                                            sparsification: SPARSIFICATION,
                                            dense: bool,
                                            k: int):
    assert directed_graph_params.grad is None

    sample = sample_graph(edge_probs=directed_graph_params,
                          undirected=undirected,
                          embeddings=embeddings,
                          sparsification=sparsification,
                          dense=dense,
                          k=k
                          )
    sample.sum().backward()
    assert directed_graph_params.grad is not None


def test_no_sparsification_has_dense_gradient(directed_graph_params: Tensor):
    assert directed_graph_params.grad is None
    sample = sparsify(edge_probs=directed_graph_params, sparsification=SPARSIFICATION.NONE)
    sample.sum().backward()

    assert directed_graph_params.grad is not None
    assert directed_graph_params.grad.nonzero(as_tuple=False).size(0) == directed_graph_params.numel()


def test_knn_sparsification_has_sparse_gradient(directed_graph_params: Tensor, embeddings: Tensor):
    assert directed_graph_params.grad is None
    sample = sparsify(edge_probs=directed_graph_params,
                      embeddings=embeddings,
                      sparsification=SPARSIFICATION.KNN,
                      k=2)
    sample.sum().backward()

    assert directed_graph_params.grad is not None
    assert directed_graph_params.grad.nonzero(as_tuple=False).size(0) != directed_graph_params.numel()


def test_knn_sparsification_with_straight_through_has_dense_gradient(directed_graph_params: Tensor,
                                                                     embeddings: Tensor):
    assert directed_graph_params.grad is None
    sample = sparsify(edge_probs=directed_graph_params,
                      embeddings=embeddings,
                      sparsification=SPARSIFICATION.KNN,
                      k=2)
    sample = straight_through_estimator(sample, directed_graph_params)
    sample.sum().backward()

    assert directed_graph_params.grad is not None
    assert directed_graph_params.grad.nonzero(as_tuple=False).size(0) == directed_graph_params.numel()


@pytest.mark.parametrize(argnames=["undirected",
                                   "sparsification",
                                   "dense",
                                   "k"],
                         argvalues=product([True, False],
                                           [SPARSIFICATION.NONE, SPARSIFICATION.KNN],
                                           [True, False],
                                           [10]))
def test_sampling_with_straight_through_has_dense_gradient(directed_graph_params: Tensor,
                                                           embeddings: Tensor,
                                                           undirected: bool,
                                                           sparsification: SPARSIFICATION,
                                                           dense: bool,
                                                           k: int,
                                                           ):
    assert directed_graph_params.grad is None

    sample = sample_graph(edge_probs=directed_graph_params,
                          undirected=undirected,
                          embeddings=embeddings,
                          sparsification=sparsification,
                          dense=dense,
                          k=k,
                          force_straight_through_estimator=True
                          )
    sample.sum().backward()
    assert directed_graph_params.grad is not None
    assert directed_graph_params.grad.nonzero(as_tuple=False).size(0) == directed_graph_params.numel()


@pytest.mark.parametrize(argnames=["undirected",
                                   "sparsification",
                                   "k",
                                   "force_straight_through_estimator"],
                         argvalues=product([True, False],
                                           [SPARSIFICATION.NONE, SPARSIFICATION.KNN],
                                           [10],
                                           [True, False]))
def test_sampling_with_sparse_input_always_has_dense_gradient(directed_graph_params: Tensor,
                                                              embeddings: Tensor,
                                                              undirected: bool,
                                                              sparsification: SPARSIFICATION,
                                                              k: int,
                                                              force_straight_through_estimator: bool
                                                              ):
    assert directed_graph_params.grad is None

    sample = sample_graph(edge_probs=directed_graph_params,
                          undirected=undirected,
                          embeddings=embeddings,
                          sparsification=sparsification,
                          dense=False,
                          k=k,
                          force_straight_through_estimator=force_straight_through_estimator
                          )
    sample.sum().backward()
    assert directed_graph_params.grad is not None
    assert directed_graph_params.grad.nonzero(as_tuple=False).size(0) == directed_graph_params.numel()


def test_normal_sampling_directed(directed_graph_params):
    graph = sample_graph(edge_probs=directed_graph_params,
                         undirected=False)
    expected = directed_graph_params
    assert graph.equal(expected)


def test_normal_sampling_undirected(directed_graph_params):
    graph = sample_graph(edge_probs=directed_graph_params,
                         undirected=True)
    expected = torch.ones((NUM_NODES, NUM_NODES)) - torch.eye(NUM_NODES)
    assert graph.equal(expected)


def test_knn_sampling_uses_numpy_dot(directed_graph_params, embeddings, monkeypatch):
    import numpy
    mocked_dot_fct = MagicMock(wraps=numpy.dot)
    monkeypatch.setattr("numpy.dot", mocked_dot_fct)
    graph = sample_graph(edge_probs=directed_graph_params,
                         embeddings=embeddings,
                         sparsification=SPARSIFICATION.KNN,
                         k=NUM_NODES - 1,
                         knn_metric="dot",
                         undirected=True)
    expected = torch.ones((NUM_NODES, NUM_NODES)) - torch.eye(NUM_NODES)
    assert mocked_dot_fct.called
    assert graph.equal(expected)


def test_knn_sampling_runs_without_error(directed_graph_params, embeddings):
    graph = sample_graph(edge_probs=directed_graph_params,
                         embeddings=embeddings,
                         sparsification=SPARSIFICATION.KNN,
                         k=NUM_NODES - 1,
                         undirected=True)


def test_knn_sampling_throws_error_without_embeddings(directed_graph_params):
    with pytest.raises(AssertionError):
        graph = sample_graph(edge_probs=directed_graph_params,
                             sparsification=SPARSIFICATION.KNN,
                             k=1,
                             undirected=True)


def test_knn_sampling_throws_error_for_k_none(directed_graph_params, embeddings):
    with pytest.raises(AssertionError):
        graph = sample_graph(edge_probs=directed_graph_params,
                             sparsification=SPARSIFICATION.KNN,
                             embeddings=embeddings,
                             k=None,
                             undirected=True)


def test_knn_sampling_throws_error_for_invalid_k(directed_graph_params, embeddings):
    with pytest.raises(AssertionError):
        graph = sample_graph(edge_probs=directed_graph_params,
                             sparsification=SPARSIFICATION.KNN,
                             embeddings=embeddings,
                             k=-5,
                             undirected=True)


def test_eps_neighborhood_sampling_does_restrict_to_correct_values(embeddings):
    eps = 0.8
    original_graph = torch.rand((NUM_NODES, NUM_NODES))
    valid_indices = (original_graph > eps).nonzero(as_tuple=False)
    graph = sparsify(edge_probs=original_graph,
                     embeddings=embeddings,
                     sparsification=SPARSIFICATION.EPS,
                     eps=eps)
    assert graph.nonzero(as_tuple=False).equal(valid_indices)
    assert torch.allclose(graph.sum(), original_graph[valid_indices[:, 0], valid_indices[:, 1]].sum())
    assert not graph.sum() == original_graph.sum()


def test_eps_neighborhood_sampling_with_low_threshold_does_restrict_to_correct_values(embeddings):
    eps = -.1
    original_graph = torch.rand((NUM_NODES, NUM_NODES))
    graph = sparsify(edge_probs=original_graph,
                     embeddings=embeddings,
                     sparsification=SPARSIFICATION.EPS,
                     eps=eps)
    assert graph.equal(original_graph)
    assert graph.sum() == original_graph.sum()


def test_eps_neighborhood_sampling_with_high_threshold_does_restrict_to_correct_values(embeddings):
    eps = 1.1
    original_graph = torch.rand((NUM_NODES, NUM_NODES))
    graph = sparsify(edge_probs=original_graph,
                     embeddings=embeddings,
                     sparsification=SPARSIFICATION.EPS,
                     eps=eps)
    assert graph.equal(torch.zeros_like(original_graph))
    assert graph.sum() == 0.0


def test_straight_through_estimator_preserves_gradients(directed_graph_params):
    assert directed_graph_params.requires_grad
    sample = Bernoulli(directed_graph_params).sample()
    straight_through_sample = straight_through_estimator(sample, directed_graph_params)
    straight_through_sample.mean().backward()
    assert sample.grad is None
    assert directed_graph_params.grad is not None
