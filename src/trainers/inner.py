from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
from higher.optim import DifferentiableOptimizer, DifferentiableAdam
from torch.optim import Adam

from src.models.gcn import MetaDenseGCN
from src.trainers import Metrics
from src.utils.evaluation import accuracy
from src.utils.graph import is_square_matrix, DenseData


def copy_detach_parameter_dict(parameters: OrderedDict) -> OrderedDict:
    _params_dict = parameters.copy()
    for key in _params_dict.keys():
        _params_dict[key] = _params_dict[key].detach().clone().requires_grad_(True)
    return _params_dict


class InnerProblemTrainer:
    def __init__(self,
                 model: MetaDenseGCN,
                 data: DenseData,
                 lr: float = 0.01,
                 weight_decay: float = 1e-4,
                 ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_params: OrderedDict = OrderedDict(model.named_parameters())
        self.optimizer: DifferentiableOptimizer = None
        self.data = data

        self.reset_optimizer()

    def reset_weights(self):
        self.model.reset_weights()
        self.model_params = OrderedDict(self.model.named_parameters())

    def reset_optimizer(self) -> None:
        optimizer = Adam(
            [
                {"params": self.model.layer_in.parameters(), "weight_decay": self.weight_decay},
                {"params": self.model.layer_out.parameters()}
            ], lr=self.lr)
        self.optimizer = DifferentiableAdam(optimizer,
                                            self.model.parameters()
                                            )

    def copy_model_params(self) -> Dict:
        return copy_detach_parameter_dict(self.model_params)

    def train_step(self,
                   graph: torch.Tensor,
                   mask: torch.Tensor = None) -> Metrics:
        """
        Does one training step with differentiable optimizer
        :param graph: Sampled graph as adjacency matrix
        :param mask: Optional, use mask other than training set
        :return: Training loss and accuracy
        """
        assert is_square_matrix(graph)

        predictions = self.model_forward(graph, is_train=True)
        mask = mask or self.data.train_mask
        loss = F.nll_loss(predictions[mask], self.data.y[mask])
        acc = accuracy(predictions[mask], self.data.y[mask])

        new_model_params = self.optimizer.step(loss, params=self.model_params.values())
        self._update_model_params(list(new_model_params))

        return Metrics(loss=loss.item(), acc=acc)

    def model_forward(self, graph, is_train: bool = True) -> torch.Tensor:
        self.model.train(mode=is_train)
        return self.model(self.data.x, graph, params=self.model_params)

    def evaluate(self,
                 graph: torch.Tensor,
                 mask: torch.Tensor = None) -> Metrics:
        """
        Calculate validation set metrics
        :param graph: Graph as adjacency matrix
        :param mask: Optional, specify mask other than validation set
        :return: Wrapper class containing loss and accuracy
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model_forward(graph, is_train=False)
            mask = mask or self.data.val_mask
            loss = F.nll_loss(predictions[mask], self.data.y[mask])
            acc = accuracy(predictions[mask], self.data.y[mask])

        return Metrics(loss=loss.item(), acc=acc)

    def detach(self) -> None:
        """
        Detach and overwrite model parameters in place to stop gradient flow. Allows for truncated backpropagation
        """
        self.model_params = copy_detach_parameter_dict(self.model_params)

        self.detach_optimizer()

    def _update_model_params(self, new_model_params: List[torch.Tensor]) -> None:
        for parameter_index, parameter_name in enumerate(self.model_params.keys()):
            self.model_params[parameter_name] = new_model_params[parameter_index]

    def detach_optimizer(self):
        """Removes all params from their compute graph in place."""
        # detach param groups
        for group in self.optimizer.param_groups:
            for k, v in group.items():
                if isinstance(v, torch.Tensor):
                    v.detach_().requires_grad_()

        # detach state
        for state_dict in self.optimizer.state:
            for k, v_dict in state_dict.items():
                if isinstance(k, torch.Tensor):
                    k.detach_().requires_grad_()
                for k2, v2 in v_dict.items():
                    if isinstance(v2, torch.Tensor):
                        v2.detach_().requires_grad_()
