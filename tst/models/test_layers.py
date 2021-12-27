import pytest
import torch

from src.models.layers import MetaDenseGraphConvolution, DenseGraphConvolution

N_FEATURES = 32
N_NODES = 5


@pytest.fixture
def batch():
    dummy_input = (torch.rand((N_NODES, N_FEATURES)))
    adj = torch.rand((N_NODES, N_NODES)).bernoulli()
    return dummy_input, adj


@pytest.mark.parametrize("model", [DenseGraphConvolution(in_features=N_FEATURES,
                                                         out_features=N_FEATURES,
                                                         use_bias=True),
                                   MetaDenseGraphConvolution(in_features=N_FEATURES,
                                                             out_features=N_FEATURES,
                                                             use_bias=True)])
def test_graph_convs_are_trainable(batch, model):
    initial_params = [param.data.clone() for param in model.parameters()]

    features, adj = batch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    out = model(features, adj)
    torch.nn.functional.mse_loss(out, out * 2).backward()
    optimizer.step()

    params_now = [param.data.clone() for param in model.parameters()]

    for a, b in zip(initial_params, params_now):
        assert not a.equal(b)


@pytest.mark.parametrize("model", [DenseGraphConvolution(in_features=N_FEATURES,
                                                         out_features=N_FEATURES,
                                                         use_bias=True),
                                   MetaDenseGraphConvolution(in_features=N_FEATURES,
                                                             out_features=N_FEATURES,
                                                             use_bias=True)])
def test_graph_convs_do_not_train(batch, model):
    initial_params = [param.data.clone() for param in model.parameters()]

    features, adj = batch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    out = model(features, adj)
    torch.nn.functional.mse_loss(out, out).backward()
    optimizer.step()

    params_now = [param.data.clone() for param in model.parameters()]

    for a, b in zip(initial_params, params_now):
        assert a.equal(b)


def test_meta_dense_graph_conv_overrides_params(batch):
    features, adj = batch
    model = MetaDenseGraphConvolution(in_features=N_FEATURES,
                                      out_features=N_FEATURES,
                                      use_bias=True)
    zero_params = dict([(name, p.clone()) for (name, p) in model.named_parameters()])
    for p in zero_params.values():
        p.data = torch.zeros_like(p.data)

    out_normal = model(features, adj)
    out_modified = model(features, adj, params=zero_params)
    assert not out_normal.equal(out_modified)
