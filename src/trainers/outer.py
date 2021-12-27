from typing import Callable, List

import torch.nn.functional as F
from sacred import Ingredient
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer

from src.models.graph import GraphGenerativeModel
from src.trainers import Metrics
from src.trainers.pretrainer import PretrainerFactory
from src.utils.evaluation import accuracy
from src.utils.graph import DenseData, graph_regularization
from src.utils.tracking import get_lr, setup_basic_logger

logger = setup_basic_logger()


class OuterProblemTrainer:

    def __init__(self,
                 optimizer: Optimizer,
                 data: DenseData,
                 opt_mask: Tensor,
                 model: GraphGenerativeModel,
                 smoothness_factor: float,
                 disconnection_factor: float,
                 sparsity_factor: float,
                 regularize: float = True,
                 lr_decay: float = None,
                 lr_decay_step_size: int = 1,
                 refine_embeddings: bool = False,
                 pretrain: bool = False,
                 ):

        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.dataset = data
        self.opt_mask = opt_mask
        self.model = model

        self.regularize = regularize
        self.smoothness_factor = smoothness_factor
        self.disconnection_factor = disconnection_factor
        self.sparsity_factor = sparsity_factor

        self.optimizer: Optimizer = optimizer
        self.lr_decayer = StepLR(self.optimizer,
                                 step_size=self.lr_decay_step_size,
                                 gamma=self.lr_decay) if self.lr_decay is not None else None

        self.refine_embeddings = refine_embeddings

        if pretrain:
            self.pretrain_model()

    def train_step(self,
                   gcn_predict_fct: Callable[[Tensor], Tensor],
                   mask: Tensor = None,
                   retain_graph: bool = True) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        graph = self.model.sample()
        predictions = gcn_predict_fct(graph)
        mask = mask or self.opt_mask
        loss = F.nll_loss(predictions[mask], self.dataset.y[mask])
        acc = accuracy(predictions[mask], self.dataset.y[mask])

        if self.regularize:
            loss += graph_regularization(graph=graph,
                                         features=self.dataset.x,
                                         smoothness_factor=self.smoothness_factor,
                                         disconnection_factor=self.disconnection_factor,
                                         sparsity_factor=self.sparsity_factor,
                                         )

        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        if self.lr_decayer is not None:
            self.lr_decayer.step(epoch=None)

        self.model.project_parameters()

        if self.refine_embeddings:
            self.model.refine()
        return Metrics(loss=loss.item(), acc=acc)

    def sample(self) -> Tensor:
        return self.model.sample()

    def detach(self):
        self.model.load_state_dict(self.model.state_dict())
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def get_learning_rates(self) -> List[float]:
        if self.optimizer is None:
            raise ValueError("Can't get optimizer learning rate, no optimizer initialized yet.")
        return get_lr(self.optimizer)

    def train(self, mode: bool = True):
        self.model.train(mode=mode)

    def eval(self):
        self.model.eval()

    def pretrain_model(self) -> None:
        pretrainer = PretrainerFactory.trainer(model=self.model, data=self.dataset)
        pretrainer.train()


class OuterProblemTrainerFactory:
    _ingredient = Ingredient("outer-trainer")

    INGREDIENTS = {
        "outer-trainer": _ingredient
    }

    @staticmethod
    @_ingredient.config
    def _config():
        lr_decay: float = 1.0
        lr_decay_step_size: int = 1
        refine_embeddings: bool = False
        pretrain: bool = True
        regularize: bool = False
        smoothness_factor: float = 0.0
        disconnection_factor: float = 0.0
        sparsity_factor: float = 0.0

    @staticmethod
    @_ingredient.capture
    def trainer(
            optimizer: Optimizer,
            data: DenseData,
            opt_mask: Tensor,
            model: GraphGenerativeModel,
            regularize: bool,
            smoothness_factor: float,
            disconnection_factor: float,
            sparsity_factor: float,
            lr_decay: float = None,
            lr_decay_step_size: int = 1,
            refine_embeddings: bool = False,
            pretrain: bool = False,
    ) -> OuterProblemTrainer:
        trainer = OuterProblemTrainer(
            optimizer=optimizer,
            data=data,
            opt_mask=opt_mask,
            model=model,
            lr_decay=lr_decay,
            lr_decay_step_size=lr_decay_step_size,
            refine_embeddings=refine_embeddings,
            pretrain=pretrain,
            regularize=regularize,
            smoothness_factor=smoothness_factor,
            disconnection_factor=disconnection_factor,
            sparsity_factor=sparsity_factor
        )
        return trainer
