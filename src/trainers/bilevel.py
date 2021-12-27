import os
from copy import deepcopy
from typing import Dict

import psutil
from sacred.run import Run

from src.trainers import Metrics
from src.trainers.inner import InnerProblemTrainer
from src.trainers.outer import OuterProblemTrainer
from src.utils.early_stopping import EarlyStopping
from src.utils.evaluation import empirical_mean_loss
from src.utils.graph import DenseData
from src.utils.tracking import setup_basic_logger


class BilevelProblemRunner:

    def __init__(self,
                 inner_trainer: InnerProblemTrainer,
                 outer_trainer: OuterProblemTrainer,
                 data: DenseData,
                 n_samples_empirical_mean: int = 16,
                 ):
        self.inner_trainer = inner_trainer
        self.outer_trainer = outer_trainer
        self.data = data
        self.gcn_params = None
        self.graph_state_dict = None
        self.n_samples_empirical_mean = n_samples_empirical_mean

        self.logger = setup_basic_logger()

    def train(self,
              patience: int,
              hyper_gradient_interval: int,
              inner_loop_max_epochs: int = 400,
              outer_loop_max_epochs: int = 400,
              sacred_runner: Run = None
              ):
        outer_early_stopper = EarlyStopping(patience=patience,
                                            max_epochs=outer_loop_max_epochs)
        current_step = 0
        outer_step = 0
        while not outer_early_stopper.abort:  # Depends on empirical mean validation loss
            inner_early_stopper = EarlyStopping(patience=patience, max_epochs=inner_loop_max_epochs)

            self.inner_trainer.reset_weights()
            self.inner_trainer.reset_optimizer()

            self.logger.info("Starting new outer loop...")

            while not inner_early_stopper.abort:  # Depends on training loss
                train_set_metrics = self.inner_opt_step()
                inner_early_stopper.update(train_set_metrics.loss,
                                           model_params=self.inner_trainer.copy_model_params())

                if sacred_runner is not None:
                    sacred_runner.log_scalar("loss.train", train_set_metrics.loss, step=current_step)
                    sacred_runner.log_scalar("acc.train", train_set_metrics.acc, step=current_step)
                    sacred_runner.log_scalar("Memory Usage (%)", psutil.Process(os.getpid()).memory_percent())

                self.logger.info(f"Model Optimization Step {current_step}: "
                                 f"loss={train_set_metrics.loss}, accuracy={train_set_metrics.acc}"
                                 )

                """
                Optimize Graph Parameters every 'hyper_gradient_interval' steps
                """
                if hyper_gradient_interval == 0 or current_step % hyper_gradient_interval == 0:
                    self.hyper_opt_step(current_step, sacred_runner)

                current_step += 1

            self.logger.info(f"Exited inner optimization")

            gcn_model_params = inner_early_stopper.model_params

            self.outer_trainer.train(False)
            empirical_val_results, empirical_test_results = \
                empirical_mean_loss(self.inner_trainer.model,
                                    graph_model=self.outer_trainer.model,
                                    n_samples=self.n_samples_empirical_mean,
                                    data=self.data,
                                    model_parameters=gcn_model_params)

            if sacred_runner is not None:
                sacred_runner.log_scalar("loss.val.empirical", empirical_val_results.loss)
                sacred_runner.log_scalar("acc.val.empirical", empirical_val_results.acc)
                sacred_runner.log_scalar("loss.test.empirical", empirical_test_results.loss)
                sacred_runner.log_scalar("acc.test.empirical", empirical_test_results.acc)

            self.logger.info(f"Empirical Validation Set Results: loss={empirical_val_results.loss}, "
                             f"accuracy={empirical_val_results.acc}")

            outer_early_stopper.update(empirical_val_results.loss,
                                       model_params=[deepcopy(gcn_model_params),
                                                     self.outer_trainer.model.state_dict()])
            outer_step += 1
        self.logger.info(f"Ended training after {outer_step} steps...")
        self.gcn_params, self.graph_state_dict = outer_early_stopper.model_params

    def inner_opt_step(self) -> Metrics:
        self.outer_trainer.train()
        graph = self.outer_trainer.sample()
        train_set_metrics = self.inner_trainer.train_step(graph)
        return train_set_metrics

    def hyper_opt_step(self, current_step: int, sacred_runner: Run = None):
        self.logger.info(f"Optimizing graph parameters at step {current_step}")
        metrics = self.outer_trainer.train_step(self.inner_trainer.model_forward)
        self.inner_trainer.detach()
        self.outer_trainer.detach()

        if sacred_runner is not None:
            sacred_runner.log_scalar("loss.outer", metrics.loss, step=current_step)
            sacred_runner.log_scalar("acc.outer", metrics.acc, step=current_step)
            for i, lr in enumerate(self.outer_trainer.get_learning_rates()):
                sacred_runner.log_scalar(f"Outer Learning Rate {i}", lr, step=current_step)
            self.logger.info(f"Graph Model Statistics:")
            for name, value in self.outer_trainer.model.statistics().items():
                sacred_runner.log_scalar(name, value, step=current_step)
                self.logger.info(f"{name}: {value}")
        self.logger.info(
            f"Performance on held-out sample for graph optimization: loss={metrics.loss}, accuracy={metrics.acc}"
            f"Outer optimizer learning rate: {self.outer_trainer.get_learning_rates()}")

    def evaluate(self) -> Dict:
        assert self.gcn_params is not None and \
               self.graph_state_dict is not None, "Models need to be trained before evaluation."
        model_params, graph_model_state_dict = self.gcn_params, self.graph_state_dict
        self.outer_trainer.model.load_state_dict(graph_model_state_dict)

        empirical_val_results, empirical_test_results = \
            empirical_mean_loss(self.inner_trainer.model,
                                graph_model=self.outer_trainer.model,
                                n_samples=self.n_samples_empirical_mean,
                                data=self.data,
                                model_parameters=model_params)
        return {
            "loss.val.final": empirical_val_results.loss,
            "acc.val.final": empirical_val_results.acc,
            "loss.test.final": empirical_test_results.loss,
            "acc.test.final": empirical_test_results.acc,
        }
