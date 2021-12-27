from typing import Iterable, Tuple, Any, Dict

import torch
import torch.nn.functional as F
from sacred import Ingredient
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import SGD, Adam
from torch_geometric.nn.models import GAE

from src.models.graph import GraphGenerativeModel, GraphProposalNetwork
from src.utils.early_stopping import EarlyStopping
from src.utils.graph import DenseData, to_dense_adj
from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()


class Pretrainer:

    def __init__(self,
                 model: GraphGenerativeModel,
                 data: DenseData,
                 lr: float,
                 optimizer: str,
                 patience: int,
                 max_epochs: int
                 ):
        self.model = model
        self.opt = self.optimizer(model, lr=lr, optimizer=optimizer)
        self.device = next(model.parameters()).device

        # Split data with pytorch geometric utility function
        gae = GAE(encoder=None)
        self.data = gae.split_edges(data.clone().to("cpu")).to(self.device)

        self.train_adj = to_dense_adj(self.data.train_pos_edge_index, num_max_nodes=self.data.num_nodes)
        self.early_stopper = EarlyStopping(patience=patience, max_epochs=max_epochs)
        self.update_model(model=model, train_adj=self.train_adj)

        logger.info(self.model)
        logger.info(self.opt)

    @staticmethod
    def update_model(model: GraphGenerativeModel,
                     train_adj: torch.Tensor):
        if type(model) == GraphProposalNetwork:
            model.adj = train_adj

    def train(self) -> Dict[str, float]:
        epoch = 0
        while not self.early_stopper.abort:
            self.train_step(epoch)
            epoch += 1

        # Reset Optimizer to original state
        self.model.load_state_dict(self.early_stopper.best_model_state_dict())
        return dict(self.evaluate(pos_index=self.data.test_pos_edge_index,
                                  neg_index=self.data.test_neg_edge_index))

    @staticmethod
    def optimizer(model: GraphGenerativeModel,
                  lr: float,
                  optimizer: str):
        assert optimizer.lower() in ["sgd", "adam"]
        opt = Adam if (optimizer.lower() == "adam") else SGD
        return opt(model.parameters(), lr=lr)

    def train_step(self, epoch: int):
        self.model.train()
        self.opt.zero_grad()
        edge_probabilities = self.model.forward()

        pos_weight = (self.train_adj.numel() - self.train_adj.sum().item()) / self.train_adj.sum().item()
        weight_matrix = (self.train_adj * (pos_weight - 1)) + 1.0

        loss = F.binary_cross_entropy(edge_probabilities,
                                      self.train_adj,
                                      weight=weight_matrix)

        loss.backward()
        self.opt.step()

        msg = f"Epoch {str(epoch).zfill(3)}: loss={loss.item()}"
        val_results = self.evaluate(pos_index=self.data.val_pos_edge_index,
                                    neg_index=self.data.val_neg_edge_index)
        for metric_name, value in val_results.items():
            msg += f" val_{metric_name}={value}"
        for metric_name, value in self.evaluate(pos_index=self.data.test_pos_edge_index,
                                                neg_index=self.data.test_neg_edge_index).items():
            msg += f" test_{metric_name}={value}"
        logger.info(f"Graph Model Statistics:")
        for name, value in self.model.statistics().items():
            logger.info(f"{name}: {value}")
        avg_prec = val_results.get("average_precision", 0.0)
        self.early_stopper.update(-avg_prec, model=self.model)
        logger.info(msg)

    def evaluate(self, pos_index: torch.Tensor, neg_index: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            edge_probabilities = self.model.forward()

        pos_pred = edge_probabilities[pos_index[0], pos_index[1]]
        neg_pred = edge_probabilities[neg_index[0], neg_index[1]]
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu()

        pos_y = torch.ones(pos_index.size(1))
        neg_y = torch.zeros(neg_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0).cpu()

        results = [(name, function(y, pred)) for name, function in zip(["auc", "average_precision"],
                                                                       [roc_auc_score, average_precision_score])]
        return dict(results)


class PretrainerFactory:
    _ingredient = Ingredient("pretrainer")
    INGREDIENTS = {
        "pretrainer": _ingredient
    }

    @staticmethod
    @_ingredient.config
    def _config():
        lr: float = 0.01
        optimizer: str = "adam"
        patience: int = 20
        max_epochs: int = 400

    @staticmethod
    @_ingredient.capture
    def trainer(model: GraphGenerativeModel,
                data: DenseData,
                lr: float,
                optimizer: str,
                patience: int,
                max_epochs: int) -> Pretrainer:
        return Pretrainer(model=model,
                          data=data,
                          lr=lr,
                          optimizer=optimizer,
                          patience=patience,
                          max_epochs=max_epochs)
