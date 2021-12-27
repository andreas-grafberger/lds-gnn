from logging import Logger
from typing import Union, Dict

import numpy as np
import torch
from sacred import Experiment
from sacred.run import Run
from seml import database_utils as db_utils, misc

from src.data.dataloader import DataFactory
from src.models.gcn import MetaDenseGCN
from src.models.factory import GraphGenerativeModelFactory
from src.trainers.bilevel import BilevelProblemRunner
from src.trainers.inner import InnerProblemTrainer
from src.trainers.outer import OuterProblemTrainerFactory
from src.trainers.pretrainer import PretrainerFactory
from src.models.sampling import Sampler
from src.utils.graph import split_mask

ingredients = list(GraphGenerativeModelFactory.INGREDIENTS.values()) + \
              list(DataFactory.INGREDIENTS.values()) + \
              list(PretrainerFactory.INGREDIENTS.values()) + \
              list(Sampler.INGREDIENTS.values()) + \
              list(OuterProblemTrainerFactory.INGREDIENTS.values())
ex = Experiment(ingredients=ingredients)
misc.setup_logger(ex)


@ex.config
def config():
    overwrite: bool = None
    db_collection: str = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))
    # from src.utils.tracking import attach_neptune_observer
    # attach_neptune_observer(ex=ex, project_name="andreas-grafberger/sandbox")


@ex.automain
def run(_run: Run,
        _log: Logger,
        _seed: int,
        device: Union[str, torch.device] = "cpu",
        hidden_sizes: int = 16,
        dropout: float = 0.5,
        gcn_optimizer_learning_rate: float = 0.01,
        gcn_weight_decay: float = 5e-4,
        graph_model: str = "lds",
        hyper_gradient_interval: int = 5,
        n_samples_empirical_mean: int = 16,
        patience: int = 20,
        ) -> Dict:
    """
    :param _run:
    :param _log:
    :param _seed:
    :param device: Name of device to run experiment on. E.g. 'cpu' or 'cuda:0'
    :param hyper_gradient_interval: Number of inner steps after which we calculate the hyper-gradient via reverse-mode
    truncated automatic differentiation
    :param gcn_weight_decay: L2-regularization used to prevent overfitting in the GCN
    :param gcn_optimizer_learning_rate: Learning rate to update the GCN's weights.
    :param dropout: Dropout probability used when training the model.
    :param hidden_sizes: Hidden dimension of GCN model, same for all hidden layers.
    :param patience Patience used for early stopping in inner and outer optimization
    :param graph_model: Method to generate the tensor. Either "lds" or "embedding"
    :param n_samples_empirical_mean: Number of sampled graphs to estimate true
    validation loss for learned tensor distribution
    :param outer_lr_decay: Factor with which the learning rate for the outer optimizer is multiplied every (outer) step
    :param pretrain: Pretrain graph model on ground-truth graph for easier start
    :return:
    """

    # Load and pre-process data
    data = DataFactory.load().to(device)

    # Split Validation set into one for tensor optimization and one for early stopping
    data.val_mask, outer_opt_mask = split_mask(data.val_mask, ratio=0.5, shuffle=True, device=device)

    graph_convolutional_network = MetaDenseGCN(data.num_features,
                                               hidden_sizes,
                                               data.num_classes,
                                               dropout=dropout).to(device)

    gcn_trainer = InnerProblemTrainer(
        model=graph_convolutional_network,
        lr=gcn_optimizer_learning_rate,
        weight_decay=gcn_weight_decay,
        data=data
    )

    graph_model_factory = GraphGenerativeModelFactory(data=data)
    graph_generator_model = graph_model_factory.create(graph_model).to(device)
    graph_generator_opt = graph_model_factory.optimizer(graph_generator_model)

    graph_generator_trainer = OuterProblemTrainerFactory.trainer(optimizer=graph_generator_opt,
                                                                 data=data,
                                                                 opt_mask=outer_opt_mask,
                                                                 model=graph_generator_model,
                                                                 )

    runner = BilevelProblemRunner(inner_trainer=gcn_trainer,
                                  outer_trainer=graph_generator_trainer,
                                  n_samples_empirical_mean=n_samples_empirical_mean,
                                  data=data
                                  )

    runner.train(patience=patience,
                 hyper_gradient_interval=hyper_gradient_interval,
                 sacred_runner=_run)

    return runner.evaluate()
