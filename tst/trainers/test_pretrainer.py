from typing import NamedTuple

import pytest
import torch
from torch.optim.optimizer import Optimizer
from torch_geometric.datasets import Planetoid

from src.data.transforms import CreateDenseAdjacencyMatrix
from src.models.graph import PairwiseEmbeddingSampler, GraphGenerativeModel, GraphProposalNetwork
from src.trainers.pretrainer import Pretrainer
from tst.test_utils import resource_folder_path

EMBEDDING_DIM = 16


@pytest.fixture()
def data():
    data = Planetoid(str(resource_folder_path() / "cora"), "cora")[0]
    return CreateDenseAdjacencyMatrix()(data)


def test_fails_on_invalid_optimizer(data):
    model = PairwiseEmbeddingSampler(data.num_nodes, embedding_dim=EMBEDDING_DIM)
    with pytest.raises(AssertionError):
        Pretrainer(
            model=model,
            data=data,
            lr=0.1,
            optimizer="adagrad",
            patience=1,
            max_epochs=10
        )


def test_gae_does_not_use_all_edges_for_training(data):
    model = GraphProposalNetwork(features=data.x,
                                 dense_adj=data.dense_adj)
    trainer = Pretrainer(
        model=model,
        data=data,
        lr=0.1,
        optimizer="adam",
        patience=1,
        max_epochs=2
    )
    trainer.train()
    assert trainer.train_adj.equal(model.adj)
    assert not data.dense_adj.equal(model.adj)


def test_embedding_model_improves(data):
    torch.manual_seed(42)
    model = PairwiseEmbeddingSampler(data.num_nodes, embedding_dim=EMBEDDING_DIM)
    trainer = Pretrainer(
        model=model,
        data=data,
        lr=0.001,
        optimizer="adam",
        patience=1,
        max_epochs=10
    )
    prev_auc = 0.0
    for i in range(10):
        trainer.train_step(i)
        curr_auc = trainer.evaluate(pos_index=trainer.data.test_pos_edge_index,
                                    neg_index=trainer.data.test_neg_edge_index).get("auc")
        assert curr_auc > prev_auc


def test_gae_model_improves(data):
    torch.manual_seed(42)
    model = GraphProposalNetwork(features=data.x,
                                 dense_adj=data.dense_adj,
                                 add_original=False)
    trainer = Pretrainer(
        model=model,
        data=data,
        lr=0.001,
        optimizer="adam",
        patience=1,
        max_epochs=10
    )
    prev_auc = 0.0
    for i in range(10):
        trainer.train_step(i)
        curr_auc = trainer.evaluate(pos_index=trainer.data.test_pos_edge_index,
                                    neg_index=trainer.data.test_neg_edge_index).get("auc")
        assert curr_auc > prev_auc
