from typing import NamedTuple, Dict
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.nn.functional import log_softmax
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch_geometric.datasets import Planetoid

from src.data.transforms import CreateDenseAdjacencyMatrix
from src.models.graph import GraphGenerativeModel, PairwiseEmbeddingSampler
from src.trainers.outer import OuterProblemTrainer
from src.utils.graph import graph_regularization
from tst.test_utils import resource_folder_path


class ModelOptimizerPair(NamedTuple):
    model: GraphGenerativeModel
    optimizer: Optimizer


@pytest.fixture
def data():
    data = Planetoid(str(resource_folder_path() / "cora"), "cora")[0]
    return CreateDenseAdjacencyMatrix()(data)


@pytest.fixture(scope='session', autouse=True)
def mocked_sampler():
    from src.models.graph import Sampler
    with patch.object(Sampler, 'sample', new=lambda *args, **kwargs: args[0]) as _fixture:
        yield _fixture


@pytest.fixture
def mock_gcn_predict_fct(data):
    def forward(graph, *args, **kwargs):
        torch.manual_seed(42)
        class_prob = torch.rand(data.num_nodes, data.y.max() + 1)
        class_prob = graph @ class_prob + graph.mean()
        return log_softmax(class_prob)

    return forward


@pytest.fixture
def model_optimizer_pair(data) -> ModelOptimizerPair:
    torch.manual_seed(42)
    model = PairwiseEmbeddingSampler(n_nodes=data.num_nodes, embedding_dim=16)
    optimizer = SGD(model.parameters(), lr=1.0)
    return ModelOptimizerPair(model, optimizer)


def init_outer_trainer(data, model_optimizer_pair, modified_params: Dict = {}):
    default_params = {
        "optimizer": model_optimizer_pair.optimizer,
        "data": data,
        "opt_mask": data.val_mask,
        "model": model_optimizer_pair.model,
        "smoothness_factor": 0.0,
        "disconnection_factor": 0.0,
        "sparsity_factor": 0.0,
        "regularize": False,
        "lr_decay": 1.0,
        "lr_decay_step_size": 1,
        "refine_embeddings": False,
        "pretrain": False
    }
    for name, val in modified_params.items():
        default_params[name] = val

    return OuterProblemTrainer(**default_params)


def test_does_pretrain(data, model_optimizer_pair):
    factory_method_mock = MagicMock()
    mocked_trainer = MagicMock()
    factory_method_mock.return_value = mocked_trainer
    with patch("src.trainers.outer.PretrainerFactory.trainer", factory_method_mock):
        trainer = init_outer_trainer(data, model_optimizer_pair, {"pretrain": True})
        assert mocked_trainer.train.called


def test_does_not_pretrain(data, model_optimizer_pair):
    mock = MagicMock()
    with patch("src.trainers.outer.PretrainerFactory.trainer", mock):
        trainer = init_outer_trainer(data, model_optimizer_pair, {"pretrain": False})
        assert not mock.called


def test_does_refine_embeddings(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair, {"refine_embeddings": True})

    with patch.object(model_optimizer_pair.model, "refine", wraps=model_optimizer_pair.model.refine) as mocked_refine:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert mocked_refine.called


def test_does_not_refine_embeddings(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair, {"refine_embeddings": False})

    with patch.object(model_optimizer_pair.model, "refine", wraps=model_optimizer_pair.model.refine) as mocked_refine:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert not mocked_refine.called


def test_does_use_train_mode_of_model(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair)

    with patch.object(model_optimizer_pair.model, "train", wraps=model_optimizer_pair.model.train) as mocked_train:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert mocked_train.called


def test_does_project_parameters(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair)

    with patch.object(model_optimizer_pair.model, "project_parameters",
                      wraps=model_optimizer_pair.model.project_parameters) as mocked_project_parameters:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert mocked_project_parameters.called


def test_calls_zero_grads_every_step(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair)

    with patch.object(model_optimizer_pair.optimizer, "zero_grad",
                      wraps=model_optimizer_pair.optimizer.zero_grad) as mocked_zero_grad:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert mocked_zero_grad.called


def test_calls_step(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair)

    with patch.object(model_optimizer_pair.optimizer, "step",
                      wraps=model_optimizer_pair.optimizer.step) as mocked_step:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        mocked_step.assert_called_once()


def test_uses_regularization(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair, {"regularize": True})

    with patch("src.trainers.outer.graph_regularization",
               wraps=graph_regularization) as mocked_graph_regularization:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        mocked_graph_regularization.assert_called_once()


def test_uses_no_regularization(data, mock_gcn_predict_fct, model_optimizer_pair):
    trainer = init_outer_trainer(data, model_optimizer_pair, {"regularize": False})

    with patch("src.trainers.outer.graph_regularization",
               wraps=graph_regularization) as mocked_graph_regularization:
        trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)
        assert not mocked_graph_regularization.called


def test_does_update_all_parameters(data, mock_gcn_predict_fct, model_optimizer_pair):
    model = model_optimizer_pair.model
    trainer = init_outer_trainer(data, model_optimizer_pair)

    initial_params = [param.data.clone() for param in model.parameters()]
    trainer.train_step(gcn_predict_fct=mock_gcn_predict_fct)

    params_now = [param.data.clone() for param in model.parameters()]

    for a, b in zip(initial_params, params_now):
        assert (a != b).all()
