from typing import Dict

import torch.nn.functional as F
from sacred import Ingredient
from sacred.run import Run
from torch.optim.optimizer import Optimizer

from src.models.gcn import MetaDenseGCN
from src.models.graph import GraphGenerativeModel
from src.trainers import Metrics
from src.utils.early_stopping import EarlyStopping
from src.utils.evaluation import accuracy, empirical_mean_loss
from src.utils.graph import DenseData, graph_regularization
from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()


class NaiveProblemRunner:
    def __init__(self,
                 gcn: MetaDenseGCN,
                 gcn_optimizer: Optimizer,
                 graph_model: GraphGenerativeModel,
                 graph_model_optimizer: Optimizer,
                 data: DenseData,
                 smoothness_factor: float,
                 disconnection_factor: float,
                 sparsity_factor: float,
                 n_samples_empirical_mean: int,
                 patience: int,
                 max_epochs: int,
                 regularize: bool = True,
                 ):
        self.data = data
        self.gcn = gcn
        self.gcn_optimizer = gcn_optimizer
        self.graph_model = graph_model
        self.graph_model_optimizer = graph_model_optimizer

        self.smoothness_factor = smoothness_factor
        self.disconnection_factor = disconnection_factor
        self.sparsity_factor = sparsity_factor
        self.regularize = regularize

        self.n_samples_empirical_mean = n_samples_empirical_mean

        self.early_stopper = EarlyStopping(patience=patience,
                                           max_epochs=max_epochs)

    def train_step(self) -> Metrics:
        self.gcn_optimizer.zero_grad()
        self.graph_model_optimizer.zero_grad()

        self.gcn.train()
        self.graph_model.train()

        graph = self.graph_model.sample()
        predictions = self.gcn.forward(self.data.x, graph, params=None)
        mask = self.data.train_mask
        acc = accuracy(predictions[mask], self.data.y[mask])
        loss = F.nll_loss(predictions[mask], self.data.y[mask])
        if self.regularize:
            loss += graph_regularization(graph=graph,
                                         features=self.data.x,
                                         smoothness_factor=self.smoothness_factor,
                                         disconnection_factor=self.disconnection_factor,
                                         sparsity_factor=self.sparsity_factor)

        loss.backward(retain_graph=True)
        self.gcn_optimizer.step()
        self.graph_model_optimizer.step()

        return Metrics(loss=loss.item(), acc=acc)

    # noinspection Mypy
    def evaluate(self) -> Dict[str, float]:

        best_gcn_state_dict, best_graph_model_state_dict = self.early_stopper.model_params

        self.gcn.load_state_dict(best_gcn_state_dict)
        self.graph_model.load_state_dict(best_graph_model_state_dict)

        empirical_val_results, empirical_test_results = \
            empirical_mean_loss(self.gcn,
                                graph_model=self.graph_model,
                                n_samples=self.n_samples_empirical_mean,
                                data=self.data)

        return {
            "loss.val.final": empirical_val_results.loss,
            "acc.val.final": empirical_val_results.acc,
            "loss.test.final": empirical_test_results.loss,
            "acc.test.final": empirical_test_results.acc,
        }

    def train(self,
              sacred_runner: Run = None,
              **kwargs):
        current_step = 0

        while not self.early_stopper.abort:

            train_metrics = self.train_step()

            if sacred_runner is not None:
                sacred_runner.log_scalar("loss.outer", train_metrics.loss, step=current_step)
                sacred_runner.log_scalar("acc.outer", train_metrics.acc, step=current_step)
                logger.info(f"Graph Model Statistics:")
                for name, value in self.graph_model.statistics().items():
                    sacred_runner.log_scalar(name, value, step=current_step)
                    logger.info(f"{name}: {value}")
            logger.info(f"Train loss={train_metrics.loss}, accuracy={train_metrics.acc}")

            empirical_val_results, _ = empirical_mean_loss(self.gcn,
                                                           graph_model=self.graph_model,
                                                           n_samples=self.n_samples_empirical_mean,
                                                           data=self.data)

            self.early_stopper.update(empirical_val_results.loss,
                                      model_params=[self.gcn.state_dict(),
                                                    self.graph_model.state_dict()])

            if sacred_runner is not None:
                sacred_runner.log_scalar("loss.val.empirical", empirical_val_results.loss)
                sacred_runner.log_scalar("acc.val.empirical", empirical_val_results.acc)

            logger.info(f"Empirical Validation Set Results: loss={empirical_val_results.loss}, "
                        f"accuracy={empirical_val_results.acc}")

            current_step += 1


class NaiveProblemRunnerFactory:
    _ingredient = Ingredient("naive-runner")

    INGREDIENTS = {
        'naive-runner': _ingredient
    }

    @staticmethod
    @_ingredient.config
    def config():
        smoothness_factor: float = 0.0
        disconnection_factor: float = 0.0
        sparsity_factor: float = 0.0
        n_samples_empirical_mean: int = 1
        patience: int = 20
        max_epochs: int = 10000
        regularize: bool = False

    @staticmethod
    @_ingredient.capture
    def runner(gcn: MetaDenseGCN,
               gcn_optimizer: Optimizer,
               graph_model: GraphGenerativeModel,
               graph_model_optimizer: Optimizer,
               data: DenseData,
               smoothness_factor: float,
               disconnection_factor: float,
               sparsity_factor: float,
               n_samples_empirical_mean: int,
               patience: int,
               max_epochs: int,
               regularize: bool) -> NaiveProblemRunner:
        runner = NaiveProblemRunner(
            gcn=gcn,
            gcn_optimizer=gcn_optimizer,
            graph_model=graph_model,
            graph_model_optimizer=graph_model_optimizer,
            data=data,
            smoothness_factor=smoothness_factor,
            disconnection_factor=disconnection_factor,
            sparsity_factor=sparsity_factor,
            n_samples_empirical_mean=n_samples_empirical_mean,
            patience=patience,
            max_epochs=max_epochs,
            regularize=regularize
        )
        return runner
