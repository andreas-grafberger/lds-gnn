from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter

from src.models.gcn import MetaDenseGCN
from src.models.sampling import Sampler
from src.utils.graph import (get_triu_values, triu_values_to_symmetric_matrix, is_square_matrix, cosine_similarity)
from src.utils.tracking import setup_basic_logger

logger = setup_basic_logger()


class ParameterClamper(object):
    def __call__(self, module):
        for param in module.parameters():
            w = param.data
            w.clamp_(0.0, 1.0)


class GraphGenerativeModel(nn.Module, ABC):

    def __init__(self, sample_undirected: bool = True, *args, **kwargs):
        super(GraphGenerativeModel, self).__init__(*args, **kwargs)
        self.sample_undirected = sample_undirected

    def sample(self, *args, **kwargs) -> Tensor:
        probs = self.forward()
        edges = Sampler.sample(probs)
        return edges

    def project_parameters(self):
        pass

    def refine(self):
        logger.warn(f"Model called to refine current parameters but method is not implemented. Ignore...")

    @abstractmethod
    def statistics(self) -> Dict[str, float]:
        pass


class BernoulliGraphModel(GraphGenerativeModel):

    def __init__(self, init_matrix: Tensor, directed: bool = False):
        """
        :param directed:
        :param init_matrix: Either symmetric matrix or flattened
            array of the values of the upper triangular matrix
        """
        super(BernoulliGraphModel, self).__init__()
        assert is_square_matrix(init_matrix)

        self.directed = directed
        self.orig_matrix = init_matrix

        # Init Values
        probs = init_matrix if directed else get_triu_values(init_matrix)
        self.probs = Parameter(probs, requires_grad=True)

    def project_parameters(self):
        self.apply(ParameterClamper())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.probs if self.directed else triu_values_to_symmetric_matrix(self.probs)  # type: ignore

    def statistics(self) -> Dict[str, float]:
        sample = self.forward()
        n_edges = sample.size(0) ** 2
        return {
            "expected_num_edges": sample.sum().item(),
            "percentage_edges_expected": sample.sum().item() / n_edges,
            "mean_prob": torch.mean(self.probs).item(),
            "min_prob": torch.min(self.probs).item(),
            "max_prob": torch.max(self.probs).item()
        }


class PairwiseEmbeddingSampler(GraphGenerativeModel):

    def __init__(self,
                 n_nodes: int,
                 embedding_dim: int,
                 prob_pow: float = 1.0,
                 init_bounds: float = 0.001):
        super(PairwiseEmbeddingSampler, self).__init__()
        self.embeddings = Parameter(torch.empty((n_nodes, embedding_dim)), requires_grad=True)
        self.prob_pow = prob_pow
        self.n_edges = n_nodes ** 2
        self.init_bounds = init_bounds

        self.reset_embeddings()

    def reset_embeddings(self):
        self.embeddings.data.uniform_(-self.init_bounds, self.init_bounds)

    def forward(self, *args, **kwargs) -> Tensor:
        return torch.sigmoid(self.embeddings @ self.embeddings.t()) ** self.prob_pow

    def sample(self, *args, **kwargs) -> Tensor:
        edge_probs = self.forward()
        edges = Sampler.sample(edge_probs, embeddings=self.embeddings)
        return edges

    def statistics(self) -> Dict[str, float]:
        probs = self.forward()
        return {
            "expected_num_edges": probs.sum().item(),
            "percentage_edges_expected": probs.sum().item() / self.n_edges
        }


class GraphProposalNetwork(GraphGenerativeModel):

    def __init__(self,
                 features: Tensor,
                 dense_adj: Tensor,
                 dropout: float = 0.0,
                 add_original: bool = False,
                 embedding_dim: int = 128,
                 probs_bias_init: float = 0.0,
                 probs_factor_init: float = 1.0,
                 prob_power: float = 1.0,
                 use_sigmoid: bool = True,
                 use_tanh: bool = False,
                 normalize_similarities: bool = False
                 ):
        super(GraphProposalNetwork, self).__init__()

        assert features.size(0) == dense_adj.size(0)
        assert is_square_matrix(dense_adj)
        assert not (use_sigmoid and use_tanh)
        assert probs_factor_init > 0.0

        self.original_features = features
        self.original_adj = dense_adj

        self.features = features
        self.adj = dense_adj
        self.n_edges = dense_adj.size(0) * dense_adj.size(1)
        self.num_features = features.size(1)
        self.add_original = add_original
        self.prob_power = prob_power
        self.use_sigmoid = use_sigmoid
        self.use_tanh = use_tanh
        self.normalize_similarities = normalize_similarities
        self.gcn = MetaDenseGCN(in_features=self.num_features,
                                hidden_features=embedding_dim * 2,
                                out_features=embedding_dim,
                                dropout=dropout)

        self.probs_factor = Parameter(torch.tensor(probs_factor_init), requires_grad=True)
        self.probs_bias = Parameter(torch.tensor(probs_bias_init), requires_grad=True)

        self.embeddings_cached = None
        self.adj_cached = None

    def forward(self,
                *args,
                return_embeddings: bool = False,
                **kwargs) -> Tensor:
        new_adj, _ = self.calculate_edges_and_embeddings()
        return new_adj

    def calculate_edges_and_embeddings(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        new_embeddings = self.gcn.forward_to_last_layer(self.features, self.adj)

        if self.normalize_similarities:
            similarity_matrix = cosine_similarity(new_embeddings, new_embeddings)
        else:
            similarity_matrix = new_embeddings @ new_embeddings.t()
        # Introduce bias for probabilities and e.g. make sigmoid steeper
        new_adj = self.probs_factor * similarity_matrix + self.probs_bias
        new_adj = torch.sigmoid(new_adj) if self.use_sigmoid else new_adj
        new_adj = torch.tanh(new_adj) if self.use_tanh else new_adj
        new_adj = new_adj + self.adj if self.add_original else new_adj
        new_adj = torch.clamp(new_adj, 0., 1.)
        return new_adj, new_embeddings

    def sample(self, *args, **kwargs) -> Tensor:
        edge_probs, embeddings = self.calculate_edges_and_embeddings()
        edges = Sampler.sample(edge_probs, embeddings=embeddings)
        self.adj_cached, self.embeddings_cached = edges, embeddings
        return edges

    def refine(self):
        if self.adj_cached is not None and self.embeddings_cached is not None:
            self.features = self.embeddings_cached
            self.adj = self.adj_cached

    def statistics(self) -> Dict[str, float]:
        probs = self.forward()
        return {
            "expected_num_edges": probs.sum().item(),
            "percentage_edges_expected": probs.sum().item() / self.n_edges,
            "probs_factor": self.probs_factor.item(),
            "probs_bias": self.probs_bias.item()
        }
