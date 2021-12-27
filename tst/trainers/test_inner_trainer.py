from typing import NamedTuple

import pytest
import torch
from torch.optim.optimizer import Optimizer
from torch_geometric.datasets import Planetoid

from src.data.transforms import CreateDenseAdjacencyMatrix
from src.models.gcn import MetaDenseGCN
from src.trainers.inner import InnerProblemTrainer
from tst.test_utils import resource_folder_path


class ModelOptimizerPair(NamedTuple):
    model: MetaDenseGCN
    optimizer: Optimizer


@pytest.fixture
def data():
    data = Planetoid(str(resource_folder_path() / "cora"), "cora")[0]
    return CreateDenseAdjacencyMatrix()(data)


@pytest.fixture
def model(data) -> MetaDenseGCN:
    torch.manual_seed(42)
    n_features = data.x.size(1)
    return MetaDenseGCN(in_features=n_features,
                        hidden_features=16,
                        out_features=int(data.y.max() + 1),
                        dropout=0.0)


def test_all_parameters_change(model, data):
    trainer = InnerProblemTrainer(model, data=data, lr=1.0)
    init_params = [p.detach().clone() for p in trainer.model_params.values()]
    trainer.train_step(data.dense_adj)
    updated_params = [p.detach().clone() for p in trainer.model_params.values()]
    for (a, b) in zip(init_params, updated_params):
        assert (a != b).all()


def test_backprop_through_time_works(model, data):
    trainer = InnerProblemTrainer(model, data=data, lr=1.0)
    graph = data.dense_adj.clone().requires_grad_(True)

    trainer.train_step(graph)
    trainer.train_step(torch.rand_like(data.dense_adj))
    trainer.train_step(torch.rand_like(data.dense_adj))

    trainer.model_forward(torch.rand_like(data.dense_adj)).sum().backward()
    assert graph.grad is not None


def test_detach_works(model, data):
    trainer = InnerProblemTrainer(model, data=data, lr=1.0)
    graph1 = data.dense_adj.clone().requires_grad_(True)
    graph2 = data.dense_adj.clone().requires_grad_(True)

    trainer.train_step(graph1)
    trainer.train_step(torch.rand_like(data.dense_adj))
    trainer.detach()
    trainer.train_step(graph2)
    trainer.train_step(torch.rand_like(data.dense_adj))

    trainer.model_forward(torch.rand_like(data.dense_adj)).sum().backward()

    assert graph1.grad is None
    assert graph2.grad is not None


def test_model_acc_improves(model, data):
    trainer = InnerProblemTrainer(model, data=data, lr=0.001)
    graph = data.dense_adj.clone()

    train_acc = 0.0
    for _ in range(10):
        train_metrics = trainer.train_step(graph)
        assert train_metrics.acc > train_acc
        train_acc = train_metrics.acc
