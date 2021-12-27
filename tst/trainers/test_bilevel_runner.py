from typing import NamedTuple
from unittest.mock import patch, MagicMock, Mock

import pytest
import torch
from pytest_mock import MockFixture
from torch import nn
from torch.optim import Optimizer, SGD

from src.data.transforms import CreateDenseAdjacencyMatrix
from src.data.utils import load_uci_dataset
from src.models.gcn import MetaDenseGCN
from src.models.graph import PairwiseEmbeddingSampler
from src.trainers.bilevel import BilevelProblemRunner
from src.trainers.inner import InnerProblemTrainer
from src.trainers.outer import OuterProblemTrainer
from tst.trainers.test_outer_trainer import init_outer_trainer


class ModelOptimizerPair(NamedTuple):
    model: nn.Module
    optimizer: Optimizer


class TrainerPair(NamedTuple):
    inner_trainer: InnerProblemTrainer
    outer_trainer: OuterProblemTrainer


@pytest.fixture(scope='session', autouse=True)
def setup():
    torch.manual_seed(42)
    with patch('src.trainers.bilevel.setup_basic_logger', return_value=MagicMock()) as _fixture:
        yield _fixture


@pytest.fixture(scope='session', autouse=True)
def mocked_sampler():
    from src.models.graph import Sampler
    with patch.object(Sampler, 'sample', new=lambda *args, **kwargs: args[0]) as _fixture:
        yield _fixture


@pytest.fixture
def data():
    data = load_uci_dataset("wine")
    return CreateDenseAdjacencyMatrix()(data)


@pytest.fixture
def graph_model_optimizer_pair(data) -> ModelOptimizerPair:
    torch.manual_seed(42)
    model = PairwiseEmbeddingSampler(n_nodes=data.num_nodes, embedding_dim=16)
    optimizer = SGD(model.parameters(), lr=1.0)
    return ModelOptimizerPair(model, optimizer)


def mocked_runner(mocker: MockFixture):
    runner = BilevelProblemRunner(inner_trainer=mocker.MagicMock(),
                                  outer_trainer=mocker.MagicMock(),
                                  data=mocker.MagicMock()
                                  )
    return runner


@pytest.fixture
def gcn(data) -> MetaDenseGCN:
    torch.manual_seed(42)
    n_features = data.x.size(1)
    return MetaDenseGCN(in_features=n_features,
                        hidden_features=16,
                        out_features=int(data.y.max() + 1),
                        dropout=0.0)


@pytest.fixture
def trainer_pair(data, gcn, graph_model_optimizer_pair):
    inner_trainer = InnerProblemTrainer(gcn, data=data, lr=0.01, weight_decay=1e-4)
    outer_trainer = init_outer_trainer(data, graph_model_optimizer_pair)
    return TrainerPair(inner_trainer, outer_trainer)


def test_does_not_evaluate_if_untrained(mocker: MockFixture):
    runner = mocked_runner(mocker)
    with pytest.raises(AssertionError):
        runner.evaluate()


def test_computation_graph_reset_after_outer_opt(mocker: MockFixture):
    runner = mocked_runner(mocker)
    runner.hyper_opt_step(1)
    runner.inner_trainer.detach.assert_called()
    runner.outer_trainer.detach.assert_called()


def test_graph_model_is_in_train_mode_for_inner_opt(mocker: MockFixture):
    runner = mocked_runner(mocker)
    runner.inner_opt_step()
    runner.outer_trainer.train.assert_called()
    runner.outer_trainer.eval.assert_not_called()


def test_is_initializable(data, trainer_pair):
    runner = BilevelProblemRunner(inner_trainer=trainer_pair.inner_trainer,
                                  outer_trainer=trainer_pair.outer_trainer,
                                  data=data)


def test_can_not_be_evaluated_without_training(data, trainer_pair):
    runner = BilevelProblemRunner(inner_trainer=trainer_pair.inner_trainer,
                                  outer_trainer=trainer_pair.outer_trainer,
                                  data=data)
    with pytest.raises(AssertionError):
        runner.evaluate()


def test_inner_opt_step_called(data, trainer_pair):
    runner = BilevelProblemRunner(inner_trainer=trainer_pair.inner_trainer,
                                  outer_trainer=trainer_pair.outer_trainer,
                                  data=data)
    with patch.object(runner, "inner_opt_step", wraps=runner.inner_opt_step) as patched_inner_opt_step:
        runner.train(patience=1, hyper_gradient_interval=0, inner_loop_max_epochs=1, outer_loop_max_epochs=1)
        assert patched_inner_opt_step.call_count == 4


def test_hyper_opt_step_called(data, trainer_pair):
    runner = BilevelProblemRunner(inner_trainer=trainer_pair.inner_trainer,
                                  outer_trainer=trainer_pair.outer_trainer,
                                  data=data)
    with patch.object(runner, "hyper_opt_step", wraps=runner.hyper_opt_step) as patched_hyper_opt_step:
        runner.train(patience=1, hyper_gradient_interval=0, inner_loop_max_epochs=0, outer_loop_max_epochs=0)
        assert patched_hyper_opt_step.call_count == 1
