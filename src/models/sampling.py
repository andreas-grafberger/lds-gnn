from enum import Enum
from typing import Optional

from sacred import Ingredient
from torch import Tensor
from torch.distributions import Bernoulli
import numpy as np

from src.data.utils import knn_graph_dense
from src.utils.graph import is_square_matrix, to_undirected


class SPARSIFICATION(Enum):
    NONE = 1
    KNN = 2
    EPS = 3


def sparsify(edge_probs: Tensor,
             sparsification: SPARSIFICATION,
             embeddings: Optional[Tensor] = None,
             k: Optional[int] = None,
             eps: Optional[float] = None,
             knn_metric: str = "cosine") -> Tensor:
    if sparsification == SPARSIFICATION.NONE:
        return edge_probs
    elif sparsification == SPARSIFICATION.KNN:
        assert embeddings is not None, "Needs embeddings to create knn graph"
        assert k is not None and 0 < k < edge_probs.size(0)
        if knn_metric == "dot":
            knn_metric = np.dot
        knn_graph = knn_graph_dense(embeddings.detach().cpu(), k=k, loop=False, metric=knn_metric).to(edge_probs.device)
        edges_not_in_knn_graph = (knn_graph == 0.0).nonzero(as_tuple=True)
        edge_probs = edge_probs.clone()
        edge_probs[edges_not_in_knn_graph] = 0.0  # Stops gradient flow through these edges!
        return edge_probs
    elif sparsification == SPARSIFICATION.EPS:
        assert eps is not None
        allowed_edge_indices = (edge_probs < eps).nonzero(as_tuple=True)
        edge_probs = edge_probs.clone()
        edge_probs[allowed_edge_indices] = 0.0  # Stops gradient flow through these edges!
        return edge_probs
    else:
        raise NotImplementedError()


def sample_graph(edge_probs: Tensor,
                 undirected: bool,
                 embeddings: Optional[Tensor] = None,
                 dense: bool = False,
                 k: Optional[int] = None,
                 sparsification: SPARSIFICATION = SPARSIFICATION.NONE,
                 force_straight_through_estimator: bool = False,
                 eps: Optional[float] = None,
                 knn_metric: str = "cosine"
                 ) -> Tensor:
    assert is_square_matrix(edge_probs)
    assert embeddings is None or edge_probs.size(0) == embeddings.size(0)

    if dense:
        sample = sparsify(edge_probs,
                          sparsification=sparsification,
                          embeddings=embeddings,
                          k=k,
                          eps=eps,
                          knn_metric=knn_metric)
    else:
        bernoulli_sample = Bernoulli(probs=edge_probs).sample()
        sample = sparsify(bernoulli_sample,
                          sparsification=sparsification,
                          embeddings=embeddings,
                          k=k,
                          eps=eps,
                          knn_metric=knn_metric)

    sample = to_undirected(sample, from_triu_only=True) if undirected else sample
    if force_straight_through_estimator or not dense:
        sample = straight_through_estimator(sample, edge_probs)
    return sample


def straight_through_estimator(sample: Tensor,
                               parameters: Tensor) -> Tensor:
    assert sample.size() == parameters.size()
    return (sample - parameters).detach() + parameters


class Sampler:
    _ingredient = Ingredient("sampler")
    INGREDIENTS = {
        "sampler": _ingredient
    }

    @staticmethod
    @_ingredient.config
    def config():
        undirected: bool = True
        k: int = 20
        eps: float = 0.9
        sparsification: str = "NONE"
        dense: bool = False
        knn_metric: str = "cosine"

    @staticmethod
    @_ingredient.capture
    def sample(edge_probs: Tensor,
               undirected: bool,
               sparsification: str,
               k: int,
               eps: float,
               embeddings: Tensor = None,
               dense: bool = False,
               knn_metric: str = "cosine"
               ) -> Tensor:
        """
        Gets square matrix with bernoulli parameters of graph distribution.
        Uses straight-through estimator to use gradient information even for not-sampled edges
        :param embeddings: When available pass raw embeddings to enable knn-graph construction
        :param k: Number of nearest neighbors to use
        :param sparsification: How to sparsify the graph. Leave empty or specify method (only 'KNN',
        'EPS' or  'NONE' supported)
        :param edge_probs: Bernoulli Parameters. Needs to be a square matrix
        :param undirected: Only use bernoulli parameters from triu matrix and ignore others

        :return:
        """
        assert sparsification in SPARSIFICATION.__members__
        sparsification_method = SPARSIFICATION[sparsification]
        sampled_graph = sample_graph(edge_probs=edge_probs,
                                     embeddings=embeddings,
                                     undirected=undirected,
                                     sparsification=sparsification_method,
                                     dense=dense,
                                     k=k,
                                     eps=eps,
                                     knn_metric=knn_metric
                                     )
        return sampled_graph
