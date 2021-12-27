from typing import cast, List

from sacred import Ingredient
from torch_geometric.transforms import NormalizeFeatures, Compose

from src.data.transforms import KNNGraph, MakeUndirected, RemoveEdges, ShuffleSplits, CreateDenseAdjacencyMatrix, \
    LargestSubgraph
from src.data.utils import (load_planetoid_dataset, Transform, load_uci_dataset, GRAPH_DATASETS, UCI_DATASETS)
from src.utils.graph import DenseData


class DataFactory:
    _data_ingredient = Ingredient("data")
    INGREDIENTS = {
        "data": _data_ingredient
    }

    @staticmethod
    @_data_ingredient.config
    def _data_config():
        dataset: str = "cora"
        remove_edges_percentage: float = 0.0
        normalize_features: bool = True
        shuffle_splits: bool = True
        make_undirected: bool = True
        nearest_neighbor_k: int = None
        use_largest_subgraph: bool = False
        split_seed: int = None
        knn_metric: str = "cosine"

    @staticmethod
    @_data_ingredient.capture
    def load(dataset: str,
             remove_edges_percentage: float,
             normalize_features: bool,
             shuffle_splits: bool,
             make_undirected: bool,
             nearest_neighbor_k: int,
             use_largest_subgraph: bool,
             knn_metric: str,
             split_seed: int):
        return load_process_dataset(dataset=dataset,
                                    remove_edges_percentage=remove_edges_percentage,
                                    normalize_features=normalize_features,
                                    shuffle_splits=shuffle_splits,
                                    make_undirected=make_undirected,
                                    nearest_neighbor_k=nearest_neighbor_k,
                                    use_largest_subgraph=use_largest_subgraph,
                                    knn_metric=knn_metric,
                                    seed=split_seed
                                    )


def load_process_dataset(dataset: str,
                         remove_edges_percentage: float,
                         normalize_features: bool,
                         shuffle_splits: bool,
                         make_undirected: bool,
                         nearest_neighbor_k: int,
                         use_largest_subgraph: bool,
                         knn_metric: str = "cosine",
                         seed: int = None) -> DenseData:
    transform_chain = create_transformations(remove_edges_percentage=remove_edges_percentage,
                                             normalize_features=normalize_features,
                                             shuffle_splits=shuffle_splits,
                                             make_undirected=make_undirected,
                                             nearest_neighbor_k=nearest_neighbor_k,
                                             use_largest_subgraph=use_largest_subgraph,
                                             knn_metric=knn_metric,
                                             seed=seed)

    if dataset in GRAPH_DATASETS:
        planetoid_dataset = load_planetoid_dataset(dataset)
        data = planetoid_dataset[0]
        # For convenience transfer a few attributes from dataset to data
        data.num_classes = planetoid_dataset.num_classes
        data.name = planetoid_dataset.name
    elif dataset in UCI_DATASETS:
        assert shuffle_splits, "shuffle_splits must be used when using UCI datasets!"
        data = load_uci_dataset(dataset)
        data.num_classes = data.y.unique().size(0)
        data.name = dataset
    else:
        raise NotImplementedError

    data = transform_chain(data)

    return cast(DenseData, data)


def create_transformations(remove_edges_percentage: float,
                           normalize_features: bool,
                           shuffle_splits: bool,
                           make_undirected: bool,
                           nearest_neighbor_k: int,
                           use_largest_subgraph: bool,
                           knn_metric: str,
                           seed: int = None):
    transforms: List[Transform] = [CreateDenseAdjacencyMatrix()]
    if normalize_features:
        transforms.append(NormalizeFeatures())
    if shuffle_splits:
        transforms.append(ShuffleSplits(seed=seed))
    if nearest_neighbor_k:
        transforms.append(KNNGraph(k=nearest_neighbor_k, loop=False, metric=knn_metric))
    if make_undirected:
        transforms.append(MakeUndirected())
    if remove_edges_percentage:
        transforms.append(RemoveEdges(remove_edges_percentage=remove_edges_percentage, seed=seed))
    if use_largest_subgraph:
        transforms.append(LargestSubgraph())
    transform_chain = Compose(transforms)
    return transform_chain
