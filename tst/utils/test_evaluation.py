from unittest.mock import MagicMock

import numpy as np
import torch

from src.models.gcn import MetaDenseGCN
from src.models.graph import GraphGenerativeModel
from src.utils.evaluation import accuracy, empirical_mean_loss, evaluate
from src.utils.graph import DenseData


def test_accuracy():
    predictions = torch.as_tensor([[0.1, 0.9, 0.0],
                                   [0.1, 0.9, 0.0],
                                   [0.0, 0.0, 1.0]])
    labels = torch.as_tensor([1, 0, 2])
    calculated_accuracy = accuracy(predictions, labels)
    assert np.allclose(calculated_accuracy, 2.0 / 3.0)


def test_empirical_mean_loss_uses_eval_model_of_models():
    graph_model = MagicMock(GraphGenerativeModel)
    gcn = MagicMock(MetaDenseGCN)

    try:
        empirical_mean_loss(gcn, graph_model, n_samples=16, data=MagicMock(DenseData), model_parameters=None)
    except AttributeError:
        pass
    finally:
        graph_model.eval.assert_called_once()
        gcn.eval.assert_called_once()


def test_evalue_uses_eval_mode_of_model(monkeypatch):
    gcn = MagicMock(MetaDenseGCN)
    no_grad_mock = MagicMock()
    monkeypatch.setattr("torch.no_grad", no_grad_mock)
    try:
        evaluate(gcn, data=MagicMock(DenseData))
    except AttributeError:
        pass
    finally:
        gcn.eval.assert_called_once()
        no_grad_mock.assert_called_once()
