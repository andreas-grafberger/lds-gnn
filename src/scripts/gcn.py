import logging
import time
from typing import Dict

import numpy as np
import torch
from sacred import Experiment
from sacred.run import Run
from seml import database_utils as db_utils
from seml import misc
from torch.nn import functional as F
from torch.optim import Adam

from src.data import DataFactory
from src.models.gcn import MetaDenseGCN
from src.utils.early_stopping import EarlyStopping
from src.utils.evaluation import accuracy, evaluate

logger = logging.getLogger(__name__)

ex = Experiment(ingredients=list(DataFactory.INGREDIENTS.values()))
ex.logger = logger
misc.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(_run: Run,
        _seed,
        device: str = 'cpu',
        learning_rate: float = 0.01,
        hidden_sizes: int = 16,
        patience: int = 10,
        weight_decay: float = 0.0005,
        epochs: int = 200,
        dropout: float = 0.5,
        normalize_adj: bool = True) -> Dict:
    torch.seed = _seed
    np.random.seed(_seed)

    logger.info(f"Using device {device}")

    # Load and pre-process data
    data = DataFactory.load().to(device)

    logger.info(
        f"Dataset Splits: {data.train_mask.sum()} train, {data.val_mask.sum()} val, {data.test_mask.sum()} test")

    gcn_model = MetaDenseGCN(data.num_features,
                             hidden_sizes,
                             data.num_classes,
                             dropout=dropout,
                             normalize_adj=normalize_adj).to(device)

    optimizer = Adam([
        {"params": gcn_model.layer_in.parameters(), "weight_decay": weight_decay},
        {"params": gcn_model.layer_out.parameters()}
    ],
        lr=learning_rate
    )

    early_stopper = EarlyStopping(patience)

    # TRAINING
    train_start_time = time.time()
    gcn_model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        gcn_model.train()
        out = gcn_model(data.x, data.dense_adj)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
        _run.log_scalar("train.loss", loss.item(), epoch)
        loss.backward()
        optimizer.step()

        metrics_dict = evaluate(gcn_model, data)
        for metric_name, val in metrics_dict.items():
            _run.log_scalar(metric_name, value=val, step=epoch)

        early_stopper.update(metrics_dict.get('val.loss'), model=gcn_model)
        if early_stopper.abort:
            break

        logger.info(f"Epoch {epoch}/{epochs}: train_loss={loss.item()}, train_acc={train_acc}. "
                    f"{list(metrics_dict.items())}"
                    )  # f"Patience={early_stopper.patience_left}")

    logger.info(f"Total training time: {time.time() - train_start_time}")
    gcn_model.load_state_dict(early_stopper.best_model_state_dict())
    results = evaluate(gcn_model, data)
    return results
