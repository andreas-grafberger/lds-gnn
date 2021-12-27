from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from src.utils.graph import DenseData
from src.models.gcn import MetaDenseGCN
from src.models.graph import GraphGenerativeModel
from src.trainers import Metrics


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculates accuracy
    :param predictions: Tensor with shape [N, C)
    :param labels: Tensor with shape [N]
    :return: float
    """
    return (torch.argmax(predictions, dim=-1) == labels).float().mean().item()


def evaluate(model: torch.nn.Module, data: DenseData, adj_matrix: torch.Tensor = None) -> Dict:
    """
    Convenience function to evaluate simple gcn gcn and calculate metrics on test/ validation set
    :return: Dictionary with validation and test accuracy/ loss
    """
    model.eval()
    with torch.no_grad():
        if adj_matrix is None:
            out = model(data.x, data.dense_adj)
        else:
            out = model(data.x, adj_matrix)

        val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])
        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()

        test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask]).item()

    return {
        'val.accuracy': val_acc,
        'val.loss': val_loss,
        'test.accuracy': test_acc,
        'test.loss': test_loss
    }


def empirical_mean_loss(gcn: MetaDenseGCN,
                        graph_model: GraphGenerativeModel,
                        n_samples: int,
                        data: Data,
                        model_parameters: OrderedDict = None) -> Tuple[Metrics, Metrics]:
    """
    Convenience function to calculate estimated loss/ accuracy for a specific graph distribution.
    :param gcn: GCN Model to use.
    :param graph_model: function to sample a new graph
    :param n_samples: Number of samples used. Higher numbers lead to better estimates
    :param data: Planetoid graph dataset_name (Cora or Citeseer) # TODO: Use InMemoryDataset if possible
    :param model_parameters: GCN parameters
    :return: Dictionary containing loss and accuracy
    """
    gcn.eval()
    graph_model.eval()
    with torch.no_grad():
        val_losses = []
        val_accuracies = []
        test_losses = []
        test_accuracies = []
        for _ in range(n_samples):
            graph = graph_model.sample()
            predictions = gcn(data.x, graph, params=model_parameters)

            val_losses.append(F.nll_loss(predictions[data.val_mask], data.y[data.val_mask]).item())
            val_accuracies.append(accuracy(predictions[data.val_mask], data.y[data.val_mask]))

            test_losses.append(F.nll_loss(predictions[data.test_mask], data.y[data.test_mask]).item())
            test_accuracies.append(accuracy(predictions[data.test_mask], data.y[data.test_mask]))

    val_metrics = Metrics(loss=np.mean(val_losses).item(), acc=np.mean(val_accuracies).item())
    test_metrics = Metrics(loss=np.mean(test_losses).item(), acc=np.mean(test_accuracies).item())
    return val_metrics, test_metrics
