from itertools import product
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torchtest import test_suite as model_test_suite

from src.models.factory import GraphGenerativeModelFactory
from src.models.graph import GraphProposalNetwork
from src.models.sampling import Sampler, sample_graph, SPARSIFICATION


@pytest.fixture
def features():
    return torch.rand((10, 5))


@pytest.fixture
def adj():
    return torch.eye(10)


@pytest.fixture
def batch(adj):
    dummy_input = dummy_output = torch.cat((adj[None, :], adj[None, :]), dim=0)
    return [dummy_input, dummy_output * 10.0]


@pytest.mark.parametrize(
    argnames=["dropout",
              "add_original",
              "embedding_dim",
              "probs_bias_init",
              "probs_factor_init",
              "prob_power",
              "use_sigmoid",
              "use_tanh",
              "normalize_similarities"],
    argvalues=product(
        [0.0, 0.5],
        [True, False],
        [16],
        [0.0],
        [1.0],
        [1.0, 10.0],
        [True, False],
        [False],
        [True, False]
    )
)
def test_passes_torchtest_suite(features, adj, batch,
                                dropout: float,
                                add_original: bool,
                                embedding_dim: int,
                                probs_bias_init: float,
                                probs_factor_init: float,
                                prob_power: float,
                                use_sigmoid: bool,
                                use_tanh: bool,
                                normalize_similarities: bool,
                                ):
    torch.manual_seed(42)
    model = GraphProposalNetwork(features=features, dense_adj=adj,
                                 dropout=dropout,
                                 add_original=add_original,
                                 embedding_dim=embedding_dim,
                                 probs_bias_init=probs_bias_init,
                                 probs_factor_init=probs_factor_init,
                                 prob_power=prob_power,
                                 use_sigmoid=use_sigmoid,
                                 use_tanh=use_tanh,
                                 normalize_similarities=normalize_similarities
                                 )
    model.register_forward_hook(lambda *args: model.project_parameters())
    optimizer = GraphGenerativeModelFactory.gae_optimizer(
        model=model,
        weights_lr=1.0,
        gcn_weight_decay=0.0,
        affine_prob_lr=1.0,
        optimizer_type="adam"
    )
    model_test_suite(model=model,
                     loss_fn=F.mse_loss,
                     optim=optimizer,
                     batch=batch,
                     output_range=(0.0 - 1e-7, 1.0 + 1e-7),
                     device="cpu",
                     test_vars_change=True,
                     test_output_range=True
                     )


@pytest.mark.parametrize(
    argnames=["dropout",
              "add_original",
              "embedding_dim",
              "probs_bias_init",
              "probs_factor_init",
              "prob_power",
              "use_sigmoid",
              "use_tanh",
              "normalize_similarities"],
    argvalues=product(
        [0.0, 0.5],
        [True, False],
        [16],
        [0.0],
        [1.0],
        [1.0, 10.0],
        [True, False],
        [False],
        [True, False]
    )
)
def test_passes_torchtest_suite_through_sampling_step(features, adj, batch,
                                                      dropout: float,
                                                      add_original: bool,
                                                      embedding_dim: int,
                                                      probs_bias_init: float,
                                                      probs_factor_init: float,
                                                      prob_power: float,
                                                      use_sigmoid: bool,
                                                      use_tanh: bool,
                                                      normalize_similarities: bool,
                                                      ):
    torch.manual_seed(42)
    model = GraphProposalNetwork(features=features, dense_adj=adj,
                                 dropout=dropout,
                                 add_original=add_original,
                                 embedding_dim=embedding_dim,
                                 probs_bias_init=probs_bias_init,
                                 probs_factor_init=probs_factor_init,
                                 prob_power=prob_power,
                                 use_sigmoid=use_sigmoid,
                                 use_tanh=use_tanh,
                                 normalize_similarities=normalize_similarities
                                 )
    model.register_forward_hook(lambda *args: model.project_parameters())
    optimizer = GraphGenerativeModelFactory.gae_optimizer(
        model=model,
        weights_lr=1.0,
        gcn_weight_decay=0.0,
        affine_prob_lr=1.0,
        optimizer_type="adam"
    )

    def patched_sample_fct(edge_probs, embeddings):
        return sample_graph(edge_probs=edge_probs,
                            embeddings=embeddings,
                            undirected=True,
                            dense=False,
                            k=5,
                            sparsification=SPARSIFICATION.KNN,
                            force_straight_through_estimator=False)

    with patch.object(Sampler, 'sample', new=patched_sample_fct):
        with patch.object(model, "forward", model.sample):
            model_test_suite(model=model,
                             loss_fn=F.mse_loss,
                             optim=optimizer,
                             batch=batch,
                             output_range=(0.0 - 1e-7, 1.0 + 1e-7),
                             device="cpu",
                             test_vars_change=True,
                             test_output_range=True
                             )


def test_affine_transform_learnable(features, adj, batch):
    model = GraphProposalNetwork(features=features, dense_adj=adj)

    model_test_suite(model=model,
                     loss_fn=F.mse_loss,
                     optim=torch.optim.Adam([
                         {"params": model.probs_bias},
                         {"params": model.probs_factor}
                     ]),
                     train_vars=[("bias", model.probs_bias),
                                 ("factor", model.probs_factor)],
                     batch=batch,
                     output_range=(0.0, 1.0),
                     device="cpu",
                     test_output_range=True
                     )


def test_corresponding_optimizer_trains_correctly(features, adj, batch):
    model = GraphProposalNetwork(features=features, dense_adj=adj)
    optimizer = GraphGenerativeModelFactory.gae_optimizer(
        model=model,
        weights_lr=1.0,
        gcn_weight_decay=0.0,
        affine_prob_lr=1.0,
        optimizer_type="adam"
    )

    model_test_suite(model=model,
                     loss_fn=F.mse_loss,
                     optim=optimizer,
                     batch=batch,
                     output_range=(0.0, 1.0 + 1e-5),
                     device="cpu",
                     test_vars_change=True,
                     test_output_range=True,
                     test_inf_vals=True,
                     test_nan_vals=True
                     )


@pytest.mark.parametrize("opt_config", [("sgd", torch.optim.SGD),
                                        ("adam", torch.optim.Adam),
                                        ("SGD", torch.optim.SGD),
                                        ("ADAM", torch.optim.Adam)])
def test_uses_correct_optimizer_type(features, adj, batch, opt_config):
    model = GraphProposalNetwork(features=features, dense_adj=adj)
    opt_name, correct_opt_type = opt_config
    optimizer = GraphGenerativeModelFactory.gae_optimizer(
        model=model,
        weights_lr=1.0,
        gcn_weight_decay=0.0,
        affine_prob_lr=1.0,
        optimizer_type=opt_name
    )

    assert type(optimizer) == correct_opt_type


def test_uses_sigmoid(features, adj, monkeypatch):
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_sigmoid=True)
    with patch('torch.sigmoid', wraps=torch.sigmoid) as wrapped_sigmoid:
        output = model(features, adj)
        assert wrapped_sigmoid.called


def test_uses_no_sigmoid(features, adj, monkeypatch):
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_sigmoid=False)
    with patch('torch.sigmoid', wraps=torch.sigmoid) as wrapped_sigmoid:
        output = model(features, adj)
        assert not wrapped_sigmoid.called


def test_uses_tanh(features, adj, monkeypatch):
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_tanh=True, use_sigmoid=False)
    with patch('torch.tanh', wraps=torch.tanh) as wrapped_tanh:
        output = model(features, adj)
        assert wrapped_tanh.called


def test_uses_no_tanh(features, adj, monkeypatch):
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_tanh=False, use_sigmoid=False)
    with patch('torch.tanh', wraps=torch.tanh) as wrapped_tanh:
        output = model(features, adj)
        assert not wrapped_tanh.called


def test_raises_error_when_using_sigmoid_and_tanh(features, adj):
    with pytest.raises(AssertionError):
        model = GraphProposalNetwork(features=features,
                                     dense_adj=adj,
                                     use_sigmoid=True,
                                     use_tanh=True)


def test_raises_error_when_probs_factor_init_smaller_than_0(features, adj):
    with pytest.raises(AssertionError):
        model = GraphProposalNetwork(features=features,
                                     dense_adj=adj,
                                     probs_factor_init=0.0
                                     )
    with pytest.raises(AssertionError):
        model = GraphProposalNetwork(features=features,
                                     dense_adj=adj,
                                     probs_factor_init=-1.0
                                     )


def test_raises_no_error_when_using_sigmoid_or_tanh(features, adj):
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_sigmoid=False, use_tanh=True)
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_sigmoid=True, use_tanh=False)
    model = GraphProposalNetwork(features=features, dense_adj=adj, use_sigmoid=False, use_tanh=False)
