from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torchtest import assert_vars_change
from torchtest import test_suite as torchtest_suite

from src.models.graph import BernoulliGraphModel
from src.models.sampling import Sampler
from src.utils.graph import is_square_matrix, to_undirected


@pytest.mark.parametrize("directed", [(True,), (False,)])
def test_bernoulli_model_returns_square_matrix(directed: bool):
    adj = torch.eye(10)
    model = BernoulliGraphModel(init_matrix=adj, directed=directed)
    params = model.forward()
    assert is_square_matrix(params)


@pytest.mark.parametrize("directed", [(True,), (False,)])
def test_bernoulli_model_trainable(directed: bool):
    adj = torch.eye(10)
    dummy_input = torch.cat((adj[None, :], adj[None, :]), dim=0)
    model = BernoulliGraphModel(init_matrix=adj, directed=directed)
    assert_vars_change(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        batch=[dummy_input, dummy_input * 10],
        device="cpu"
    )


@pytest.mark.parametrize("directed", [(True,), (False,)])
def test_bernoulli_model_passes_test_suite(directed: bool):
    adj = torch.eye(10)
    dummy_input = torch.cat((adj[None, :], adj[None, :]), dim=0)
    model = BernoulliGraphModel(init_matrix=adj, directed=directed)
    model.register_forward_hook(lambda *args: model.project_parameters())
    torchtest_suite(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        batch=[dummy_input, dummy_input * 10],
        device="cpu",
        output_range=(0.0 - 1e-6, 1.0 + 1e-6),
        test_output_range=True,
        test_inf_vals=True,
        test_vars_change=True,
        test_nan_vals=True
    )


def test_undirected_bernoulli_model_returns_symmetric_matrix():
    adj = torch.eye(10)
    adj[1, :] = 1.0
    adj = to_undirected(adj)

    model = BernoulliGraphModel(init_matrix=adj, directed=False)
    expected_output = torch.clamp(adj + adj.t(), 0.0, 1.0)
    parameters = model.forward()
    assert torch.equal(parameters, expected_output)


def test_undirected_bernoulli_model_gradients_flow_through_forward():
    adj = torch.eye(10)
    adj[1, :] = 1.0
    model = BernoulliGraphModel(init_matrix=adj, directed=False)
    assert model.probs.grad is None
    output = model.forward()
    output.sum().backward()
    assert model.probs.grad is not None
    assert (model.probs.grad > 0).all()


def test_directed_bernoulli_model_gradients_flow_through_forward():
    adj = torch.eye(10)
    adj[1, :] = 1.0
    model = BernoulliGraphModel(init_matrix=adj, directed=True)
    assert model.probs.grad is None
    output = model.forward()
    output.sum().backward()
    assert model.probs.grad is not None
    assert (model.probs.grad > 0).all()


def test_directed_bernoulli_model_gradients_flow_through_sampling():
    adj = torch.eye(10)
    adj[1, :] = 1.0
    model = BernoulliGraphModel(init_matrix=adj, directed=True)
    assert model.probs.grad is None
    with mock.patch.object(Sampler, 'sample', new=lambda *args, **kwargs: args[0]):
        output = model.sample()
        output.sum().backward()
        assert model.probs.grad is not None
        assert (model.probs.grad > 0).all()


def test_undirected_bernoulli_model_gradients_flow_through_sampling():
    adj = torch.eye(10)
    adj[1, :] = 1.0
    model = BernoulliGraphModel(init_matrix=adj, directed=False)
    assert model.probs.grad is None
    with mock.patch.object(Sampler, 'sample', new=lambda *args, **kwargs: args[0]):
        output = model.sample()
        output.sum().backward()
        assert model.probs.grad is not None
        assert (model.probs.grad > 0).all()


def test_directed_bernoulli_model_returns_asymmetric_matrix():
    adj = torch.eye(10)
    adj[1, :] = 1.0

    model = BernoulliGraphModel(init_matrix=adj, directed=True)
    expected_output = adj
    parameters = model.forward()
    assert torch.equal(parameters, expected_output)


def test_parameters_are_projected_correctly():
    adj = torch.eye(10) * 2
    model = BernoulliGraphModel(init_matrix=adj, directed=True)
    assert (model.forward() > 1.0).any()
    model.project_parameters()
    assert not (model.forward() > 1.0).any()
