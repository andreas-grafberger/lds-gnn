from logging import Logger
from typing import Union, Dict

import numpy as np
import torch
from sacred import Experiment
from sacred.run import Run
from seml import misc

from seml import database_utils as db_utils

from src.data import DataFactory
from src.models.factory import GraphGenerativeModelFactory
from src.trainers.pretrainer import Pretrainer, PretrainerFactory

ingredients = list(GraphGenerativeModelFactory.INGREDIENTS.values()) \
              + list(DataFactory.INGREDIENTS.values()) \
              + list(PretrainerFactory.INGREDIENTS.values())
ex = Experiment(ingredients=ingredients)
misc.setup_logger(ex)


@ex.config
def config():
    overwrite: bool = None
    db_collection: str = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(_run: Run,
        _log: Logger,
        _seed: int,
        device: Union[str, torch.device] = "cpu",
        graph_model: str = "gae",
        ) -> Dict:
    """
    :param _run:
    :param _log:
    :param _seed:
    :param device: Name of device to run experiment on. E.g. 'cpu' or 'cuda:0'
    :param graph_model: Method to generate the tensor. Either "lds" or "embedding"
    :return:
    """

    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # Load and pre-process data
    data = DataFactory.load().to(device)

    graph_model_factory = GraphGenerativeModelFactory(data=data)
    graph_generator_model = graph_model_factory.create(graph_model).to(device)

    pretrainer = PretrainerFactory.trainer(model=graph_generator_model, data=data)

    return pretrainer.train()
