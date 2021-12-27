import pytest
import torch

import torch.nn.functional as F
from torchtest import test_suite as torchtest_suite

from src.models.factory import GraphGenerativeModelFactory
from src.models.graph import PairwiseEmbeddingSampler


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


@pytest.mark.parametrize("embedding_dim", [1, 8, 16, 64, 128])
def test_passes_torchtest_suite(features, adj, batch, embedding_dim):
    model = PairwiseEmbeddingSampler(n_nodes=10, embedding_dim=embedding_dim)
    optimizer = GraphGenerativeModelFactory.embeddings_optimizer(model, lr=1.0)
    torchtest_suite(model=model,
                    loss_fn=F.mse_loss,
                    optim=optimizer,
                    batch=batch,
                    output_range=(0.0, 1.0),
                    device="cpu",
                    test_output_range=True,
                    test_vars_change=True
                    )
