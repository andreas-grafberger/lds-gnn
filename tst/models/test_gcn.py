from unittest.mock import patch

import pytest
import torch

from src.models.gcn import MetaDenseGCN
from src.utils.graph import normalize_adjacency_matrix

N_FEATURES = 5
N_NODES = 10


@pytest.fixture
def features():
    return torch.rand((N_NODES, N_FEATURES))


@pytest.fixture
def adj():
    return torch.eye(N_NODES)


@pytest.fixture
def batch():
    torch.manual_seed(3141592)
    dummy_input = (torch.rand((N_NODES, N_FEATURES)))
    adj = torch.rand((N_NODES, N_NODES)).bernoulli()
    return dummy_input, adj


def test_meta_dense_gcn_trainable(batch):
    model = MetaDenseGCN(in_features=N_FEATURES,
                         hidden_features=N_FEATURES // 2,
                         out_features=N_FEATURES,
                         dropout=0.5,
                         normalize_adj=True)
    features, adj = batch
    initial_params = [param.data.clone() for param in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    out = model(features, adj)
    torch.nn.functional.mse_loss(out, out * 2).backward()
    optimizer.step()

    params_now = [param.data.clone() for param in model.parameters()]

    for a, b in zip(initial_params, params_now):
        assert (a != b).all()


@patch('src.models.gcn.normalize_adjacency_matrix', wraps=normalize_adjacency_matrix)
def test_meta_dense_gcn_normalizes_adj(mock, batch):
    features, adj = batch
    model = MetaDenseGCN(in_features=N_FEATURES,
                         hidden_features=N_FEATURES // 2,
                         out_features=N_FEATURES,
                         dropout=0.5,
                         normalize_adj=True)
    out = model(features, adj)
    mock.assert_called()


@patch('src.models.gcn.normalize_adjacency_matrix', wraps=normalize_adjacency_matrix)
def test_meta_dense_gcn_does_not_normalize_adj(mock, batch):
    features, adj = batch
    model = MetaDenseGCN(in_features=N_FEATURES,
                         hidden_features=N_FEATURES // 2,
                         out_features=N_FEATURES,
                         dropout=0.5,
                         normalize_adj=False)
    out = model(features, adj)
    mock.assert_not_called()


def test_meta_dense_gcn_overrides_params(batch):
    model = MetaDenseGCN(in_features=N_FEATURES,
                         hidden_features=N_FEATURES // 2,
                         out_features=N_FEATURES,
                         dropout=0.5,
                         normalize_adj=True)
    features, adj = batch
    zero_params = dict([(name, p.clone()) for (name, p) in model.named_parameters()])
    for p in zero_params.values():
        p.data = torch.zeros_like(p.data)
    out_normal = model(features, adj)
    out_modified = model(features, adj, params=zero_params)
    assert not out_normal.equal(out_modified)


def test_meta_dense_gcn_overriden_params_trainable(batch):
    model = MetaDenseGCN(in_features=N_FEATURES,
                         hidden_features=N_FEATURES // 2,
                         out_features=N_FEATURES,
                         dropout=0.5,
                         normalize_adj=True)
    features, adj = batch
    zero_params = dict([(name, p.clone().detach().requires_grad_(True)) for (name, p) in model.named_parameters()])
    for p in zero_params.values():
        p.data = torch.rand_like(p.data)
    for p in zero_params.values():
        assert p.grad is None

    out_modified = model(features, adj, params=zero_params)
    out_modified.sum().backward()

    for original_param in model.parameters():
        assert original_param.grad is None
    for param in zero_params.values():
        assert param.grad is not None
