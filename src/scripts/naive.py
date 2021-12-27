from logging import Logger
from typing import Union, Dict

import numpy as np
import torch
from sacred import Experiment
from sacred.run import Run
from seml import database_utils as db_utils, misc
from torch.optim import Adam

from src.data.dataloader import DataFactory
from src.models.factory import GraphGenerativeModelFactory
from src.models.gcn import MetaDenseGCN
from src.models.sampling import Sampler
from src.trainers.naive import NaiveProblemRunnerFactory
from src.trainers.pretrainer import PretrainerFactory

ingredients = list(GraphGenerativeModelFactory.INGREDIENTS.values()) + \
              list(DataFactory.INGREDIENTS.values()) + \
              list(PretrainerFactory.INGREDIENTS.values()) + \
              list(Sampler.INGREDIENTS.values()) + \
              list(NaiveProblemRunnerFactory.INGREDIENTS.values())
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
        graph_model: str = "gae",
        ) -> Dict:
    """
    :param _run:
    :param _log:
    :param _seed:
    :param device: Name of device to run experiment on. E.g. 'cpu' or 'cuda:0'
    :param gcn_weight_decay: L2-regularization used to prevent overfitting in the GCN
    :param gcn_optimizer_learning_rate: Learning rate to update the GCN's weights.
    :param dropout: Dropout probability used when training the model.
    :param hidden_sizes: Hidden dimension of GCN model, same for all hidden layers.
    :param graph_model: Method to generate the tensor. Either "lds" or "embedding"
    validation loss for learned tensor distribution
    :return:
    """

    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # Load and pre-process data
    data = DataFactory.load().to(device)

    gcn = MetaDenseGCN(data.num_features,
                       hidden_sizes,
                       data.num_classes,
                       dropout=dropout).to(device)

    gcn_optimizer = Adam([
        {"params": gcn.layer_in.parameters(), "weight_decay": gcn_weight_decay},
        {"params": gcn.layer_out.parameters()}
    ],
        lr=gcn_optimizer_learning_rate
    )

    graph_model_factory = GraphGenerativeModelFactory(data=data)
    graph_generator_model = graph_model_factory.create(graph_model).to(device)
    graph_generator_opt = graph_model_factory.optimizer(graph_generator_model)

    runner = NaiveProblemRunnerFactory.runner(gcn=gcn,
                                              gcn_optimizer=gcn_optimizer,
                                              graph_model=graph_generator_model,
                                              graph_model_optimizer=graph_generator_opt,
                                              data=data
                                              )

    runner.train(sacred_runner=_run)

    return runner.evaluate()
