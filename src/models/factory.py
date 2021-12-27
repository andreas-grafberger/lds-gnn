from typing import Type

from sacred import Ingredient
from torch.optim import Optimizer, SGD, Adam

from src.models.graph import GraphGenerativeModel, BernoulliGraphModel, PairwiseEmbeddingSampler, GraphProposalNetwork
from src.utils.graph import DenseData


class GraphGenerativeModelFactory:
    _lds_ingredient = Ingredient("lds")
    _emb_ingredient = Ingredient("embedding")
    _gae_ingredient = Ingredient("gae")

    INGREDIENTS = {
        "lds": _lds_ingredient,
        "embedding": _emb_ingredient,
        "gae": _gae_ingredient
    }

    def __init__(self, data: DenseData):
        self.data = data

    def create(self, model_name: str) -> GraphGenerativeModel:
        model_name = model_name.lower()
        if model_name == "lds":
            model = self.lds(data=self.data)
        elif model_name == "embedding":
            model = self.embeddings(data=self.data)
        elif model_name == "gae":
            model = self.gae(data=self.data)
        else:
            raise NotImplementedError(f"Model {model_name} not supported.")
        return model  # type: ignore

    def optimizer(self, model: GraphGenerativeModel) -> Optimizer:
        model_type = type(model)
        if model_type == BernoulliGraphModel:
            opt = self.lds_optimizer(model=model)
        elif model_type == PairwiseEmbeddingSampler:
            opt = self.embeddings_optimizer(model=model)
        elif model_type == GraphProposalNetwork:
            opt = self.gae_optimizer(model=model)
        else:
            raise NotImplementedError(f"Optimizer for model type {model_type} not implemented.")
        return opt  # type: ignore

    #############
    # LDS MODEL #
    #############

    @staticmethod
    @_lds_ingredient.config
    def _lds_config():
        directed: bool = False
        lr: float = 1.0

    @staticmethod
    @_lds_ingredient.capture
    def lds(data: DenseData,
            directed: bool) -> BernoulliGraphModel:
        return BernoulliGraphModel(data.dense_adj, directed=directed)

    @staticmethod
    @_lds_ingredient.capture
    def lds_optimizer(model: BernoulliGraphModel,
                      lr: float) -> Optimizer:
        optimizer = SGD(model.parameters(), lr=lr)
        return optimizer

    ###################
    # EMBEDDING MODEL #
    ###################

    @staticmethod
    @_emb_ingredient.config
    def _embeddings_config():
        embedding_dim: int = 16
        prob_pow: float = 1.0
        lr: float = 0.1
        init_bounds: float = 0.001

    @staticmethod
    @_emb_ingredient.capture
    def embeddings(
            data: DenseData,
            embedding_dim: int,
            prob_pow: float,
            init_bounds: float
    ) -> PairwiseEmbeddingSampler:
        return PairwiseEmbeddingSampler(n_nodes=data.num_nodes,
                                        embedding_dim=embedding_dim,
                                        prob_pow=prob_pow,
                                        init_bounds=init_bounds)

    @staticmethod
    @_emb_ingredient.capture
    def embeddings_optimizer(model: PairwiseEmbeddingSampler,
                             lr: float) -> Optimizer:
        optimizer = SGD(model.parameters(), lr=lr)
        return optimizer

    #############
    # GAE MODEL #
    #############

    @staticmethod
    @_gae_ingredient.config
    def _gae_config():
        dropout: float = 0.0
        add_original: bool = False
        embedding_dim: int = 16
        probs_bias_init: float = 0.0
        probs_factor_init: float = 1.0
        prob_power: float = 1.0
        use_sigmoid: bool = True
        normalize_similarities: bool = True
        weights_lr: float = 0.01
        gcn_weight_decay: float = 0.0005
        affine_prob_lr: float = 0.01
        optimizer_type: str = "SGD"
        use_tanh: bool = False

    @staticmethod
    @_gae_ingredient.capture
    def gae(data: DenseData,
            dropout: float,
            add_original: bool,
            embedding_dim: int,
            probs_bias_init: float,
            probs_factor_init: float,
            prob_power: float,
            use_sigmoid: bool,
            use_tanh: bool,
            normalize_similarities: bool) -> GraphProposalNetwork:
        return GraphProposalNetwork(features=data.x,
                                    dense_adj=data.dense_adj,
                                    dropout=dropout,
                                    add_original=add_original,
                                    embedding_dim=embedding_dim,
                                    probs_bias_init=probs_bias_init,
                                    probs_factor_init=probs_factor_init,
                                    prob_power=prob_power,
                                    use_sigmoid=use_sigmoid,
                                    use_tanh=use_tanh,
                                    normalize_similarities=normalize_similarities
                                    )

    @staticmethod
    @_gae_ingredient.capture
    def gae_optimizer(model: GraphProposalNetwork,
                      weights_lr: float,
                      gcn_weight_decay: float,
                      affine_prob_lr: float,
                      optimizer_type: str,
                      ) -> Optimizer:
        opt_type = GraphGenerativeModelFactory.get_optimizer(optimizer_type)
        affine_prob_lr = affine_prob_lr or weights_lr
        optimizer = opt_type(
            params=[
                {
                    "params": model.gcn.parameters(),
                    "weight_decay": gcn_weight_decay,
                    "lr": weights_lr},
                {
                    "params": [model.probs_factor, model.probs_bias],
                    "lr": affine_prob_lr
                }
            ]
        )
        return optimizer

    @staticmethod
    def get_optimizer(optimizer_type) -> Type[Optimizer]:
        if optimizer_type.lower() == "sgd":
            return SGD
        elif optimizer_type.lower() == "adam":
            return Adam
        else:
            raise NotImplementedError()
