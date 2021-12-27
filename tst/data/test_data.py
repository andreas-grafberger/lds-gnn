from itertools import product
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
from torch_geometric.utils import to_undirected as edge_index_to_undirected

from src.data.dataloader import load_process_dataset
from src.data.utils import *
from src.utils.graph import to_dense_adj, to_undirected, get_triu_values
from tst.test_utils import cora, citeseer, resource_folder_path


@pytest.mark.parametrize("dataset_name", ["cora", "citeseer"])
def test_load_planetoid_dataset_applies_no_normalization(dataset_name):
    data = load_planetoid_dataset(dataset=dataset_name, path=resource_folder_path() / dataset_name)[0]
    # Remove nodes with zero features only
    data.x = data.x[data.x.sum(dim=1).nonzero(), :].squeeze()
    assert not data.x.sum(dim=1).allclose(torch.as_tensor(1.0))


def test_load_dataset(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data1 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=41,
                                     )


def test_dense_adj_to_edge_index_on_cora(cora):
    data = cora[0]
    dense_adj = to_dense_adj(data.edge_index, num_max_nodes=data.num_nodes)
    edge_index_rec = dense_adj_to_edge_index(dense_adj)
    assert torch.equal(data.edge_index, edge_index_rec)


def test_dense_adj_to_edge_index_on_dummy_data():
    dense_adj = torch.eye(5)
    dense_adj[1, :] = 1.0
    expected_edge_index = torch.as_tensor(
        [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 2], [3, 3], [4, 4], ]).t().long()
    computed_result = dense_adj_to_edge_index(dense_adj)
    assert computed_result.equal(expected_edge_index)


def test_cora_is_undirected(cora):
    data = cora[0]
    assert data.is_undirected()
    assert torch.equal(data.edge_index, edge_index_to_undirected(data.edge_index))


def test_citeseer_is_undirected(citeseer):
    data = citeseer[0]
    assert data.is_undirected()
    assert torch.equal(data.edge_index, edge_index_to_undirected(data.edge_index))


# noinspection PyArgumentList
def test_unique_edges_correct():
    edge_index = torch.LongTensor([[1, 2], [2, 3], [3, 2], [2, 3]]).t()
    calc_unique_edges = unique_edges(edge_index)
    expected_result = torch.LongTensor([[1, 2], [2, 3], [3, 2]]).t()
    assert calc_unique_edges.equal(expected_result)


def test_unique_edges_does_not_fail_on_empty_edge_input():
    # noinspection PyArgumentList
    edge_index = torch.LongTensor([[], []])
    calc_unique_edges = unique_edges(edge_index)
    assert calc_unique_edges.equal(edge_index)


def test_remove_edges_from_undirected_graph():
    m = np.eye(10)
    m[2] = 1.0
    m[:, 2] = 1.0

    dense_adj = torch.as_tensor(m).float()
    dense_adj_removed = remove_edges_from_undirected_graph(dense_adj, 1.0)

    assert (dense_adj_removed.sum() == 0.0) and \
           (torch.equal(dense_adj_removed, torch.zeros(dense_adj.size())))

    dense_adj_removed = remove_edges_from_undirected_graph(dense_adj, 0.0)
    assert torch.equal(dense_adj_removed, dense_adj)

    dense_adj = torch.as_tensor(np.eye(2)).float()
    dense_adj_removed = remove_edges_from_undirected_graph(dense_adj, 0.5)
    assert dense_adj_removed.sum() == 1.0


def test_filter_edges():
    # noinspection PyArgumentList
    edge_index = torch.LongTensor([[0, 0, 1, 2, 4, 3], [0, 1, 2, 2, 5, 2]])
    nodes_to_keep = [1, 2]

    computed_result = filter_edges(edge_index=edge_index, nodes_to_keep=nodes_to_keep)
    expected_result = np.array([
        [0, 1, 2, 3],
        [1, 2, 2, 2]
    ])
    assert (computed_result.numpy() == expected_result).all()


def test_filter_edges_allows_all():
    # noinspection PyArgumentList
    edge_index = torch.LongTensor([[0, 0, 1, 2, 3, 4], [0, 1, 2, 2, 2, 5]])
    nodes_to_keep = [0, 1, 2, 3, 4]

    computed_result = filter_edges(edge_index=edge_index, nodes_to_keep=nodes_to_keep)
    expected_result = edge_index
    assert computed_result.equal(expected_result)


def test_filter_edges_returns_empty_when_no_nodes_allowed():
    # noinspection PyArgumentList
    edge_index = torch.LongTensor([[0, 0, 1, 2, 4, 3], [0, 1, 2, 2, 5, 2]])
    computed_result = filter_edges(edge_index=edge_index, nodes_to_keep=[])
    expected_result = np.array([[], []])
    assert (computed_result.numpy() == expected_result).all()


def test_largest_connected_component_for_cora_correct(cora):
    data = cora[0]
    new_edge_index = largest_subgraph(data.edge_index)
    assert new_edge_index.size(1) < data.edge_index.size(1)
    assert torch.unique(new_edge_index).size(0) == 2485  # Same as in pitfalls paper


def test_largest_connected_component_for_citeseer_correct(citeseer):
    data = citeseer[0]
    new_edge_index = largest_subgraph(data.edge_index)
    assert new_edge_index.size(1) < data.edge_index.size(1)
    # assert torch.unique(new_edge_index).size(0) == 2110  # Same as in pitfalls paper TODO: Investigate why 2120


def test_indices_to_mask_correct():
    indices = torch.as_tensor([1, 2, 5, 7]).long()
    size = 10
    # noinspection PyTypeChecker
    calculated_result = indices_to_mask(indices, size)
    expected_result = torch.as_tensor([False, True, True, False, False, True, False, True, False, False]).bool()
    assert calculated_result.equal(expected_result)


def test_indices_to_mask_not_equal_to_wrong_result():
    indices = torch.as_tensor([1, 2, 5, 7]).long()
    size = 10
    # noinspection PyTypeChecker
    calculated_result = indices_to_mask(indices, size)
    expected_result = torch.as_tensor([False, True, False, False, False, True, False, True, False, False]).bool()
    assert not calculated_result.equal(expected_result)


def test_shuffled_splits_do_not_overlap(cora):
    splits_1, splits_2 = create_splits_for_dataset(cora, 42, 45)
    mask1 = splits_1.train_mask + splits_1.val_mask + splits_1.test_mask
    mask2 = splits_2.train_mask + splits_2.val_mask + splits_2.test_mask
    assert mask1.sum() == mask2.sum()
    assert mask1.sum() == (splits_1.train_mask.sum() + splits_1.val_mask.sum() + splits_1.test_mask.sum())


def test_indices_to_mask_fails_on_invalid_size():
    with pytest.raises(IndexError):
        indices = torch.as_tensor([1, 2, 5, 7]).long()
        calculated_result = indices_to_mask(indices, size=5)


def test_random_splits_not_equal_with_different_or_no_seeds(cora):
    splits_1, splits_2 = create_splits_for_dataset(cora, 42, 41)
    assert not datasets_have_same_splits(splits_1, splits_2)


def test_random_splits_equal_with_same_seed(cora):
    splits_1, splits_2 = create_splits_for_dataset(cora, 42, 42)
    assert datasets_have_same_splits(splits_1, splits_2)


def test_load_function_returns_equal_splits_when_planetoid_splits_are_used(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data1 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=False,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True
                                     )
        data2 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=False,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True
                                     )
        assert datasets_have_same_splits(data1, data2)


def test_load_function_returns_unequal_splits_for_random_splits_splits_without_seeds(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data1 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=None,
                                     )
        data2 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=None
                                     )
        assert not datasets_have_same_splits(data1, data2)


def test_load_function_returns_unequal_splits_for_random_splits_splits_with_different_seeds(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data1 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=41,
                                     )
        data2 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=314
                                     )
        assert not datasets_have_same_splits(data1, data2)


def test_load_function_returns_equal_splits_for_random_splits_splits_with_same_seeds(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data1 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=42
                                     )
        data2 = load_process_dataset(dataset="cora",
                                     remove_edges_percentage=.5,
                                     normalize_features=True,
                                     shuffle_splits=True,
                                     make_undirected=True,
                                     nearest_neighbor_k=10,
                                     use_largest_subgraph=True,
                                     seed=42
                                     )
        assert datasets_have_same_splits(data1, data2)


def datasets_have_same_splits(data1: Data, data2: Data) -> bool:
    splits_1 = [data1.train_mask, data1.val_mask, data1.test_mask]
    splits_2 = [data2.train_mask, data2.val_mask, data2.test_mask]

    are_equal_list: List[bool] = []
    for i, split in enumerate(splits_1):
        are_equal_list.append(split.equal(splits_2[i]))

    # Check if some are equal and some not
    assert len(set(are_equal_list)) == 1
    return are_equal_list[0]


def create_splits_for_dataset(cora, seed1, seed2) -> Tuple[Data, Data]:
    dataset = cora[0]
    dataset_1 = dataset.clone()
    shuffle_splits_(dataset_1, seed=seed1)

    dataset_2 = dataset.clone()
    shuffle_splits_(dataset_2, seed=seed2)

    return dataset_1, dataset_2


def test_knn_graph_returns_correct_indices():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=20, loop=True)
    calc_indices = knn_graph(features, k=20, loop=True)
    assert calc_adj.nonzero().t().equal(calc_indices)


def test_knn_graph_dense_returns_identity():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=1, loop=True)
    assert calc_adj.equal(torch.eye(500))


def test_knn_graph_dense_returns_fully_connected_graph():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=500, loop=True)
    assert calc_adj.equal(torch.ones_like(calc_adj))


def test_knn_graph_dense_does_contain_loops():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=10, loop=True)
    assert calc_adj.trace() == 500


def test_knn_graph_dense_does_not_contain_loops():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=10, loop=False)
    assert calc_adj.trace() == 0.0


def test_knn_graph_dense_is_directed():
    features = torch.randn((500, 128))
    calc_adj = knn_graph_dense(features, k=10)
    assert not calc_adj.equal(calc_adj.t())


def test_seed_overwrite_returns_to_correct_state():
    torch.manual_seed(5125122123123)
    old_random_state = torch.random.get_rng_state()

    with PytorchSeedOverwrite(42):
        perm = torch.randperm(5)
        assert not torch.random.get_rng_state().equal(old_random_state)

    assert torch.random.get_rng_state().equal(old_random_state)


def test_seed_overwrite_uses_root_seed_if_none_specified():
    torch.manual_seed(5125122123123)
    old_random_state = torch.random.get_rng_state()

    with PytorchSeedOverwrite():
        first_randperm = torch.randperm(100)

    with PytorchSeedOverwrite(None):
        second_randperm = torch.randperm(100)

    assert first_randperm.equal(second_randperm)
    assert torch.random.get_rng_state().equal(old_random_state)


def test_seed_overwrite_uses_specific_seed_if_specified():
    torch.manual_seed(5125122123123)
    old_random_state = torch.random.get_rng_state()

    with PytorchSeedOverwrite(42):
        first_randperm = torch.randperm(100)

    with PytorchSeedOverwrite(42):
        second_randperm = torch.randperm(100)

    assert first_randperm.equal(second_randperm)
    assert torch.random.get_rng_state().equal(old_random_state)


def test_seed_overwrite_uses_different_seed_if_specified():
    torch.manual_seed(5125122123123)
    old_random_state = torch.random.get_rng_state()

    with PytorchSeedOverwrite(42):
        first_randperm = torch.randperm(100)

    with PytorchSeedOverwrite(312):
        second_randperm = torch.randperm(100)

    assert not first_randperm.equal(second_randperm)
    assert torch.random.get_rng_state().equal(old_random_state)


def test_remove_edges_calls_directed_method(monkeypatch):
    adj = torch.rand((10, 10))

    mock = MagicMock()
    monkeypatch.setattr("src.data.utils.remove_edges_from_directed_graph", mock)

    remove_edges(adj, is_directed=True, remove_edges_percentage=0.5, seed=42)
    mock.assert_called_once()


def test_remove_edges_calls_undirected_method(monkeypatch):
    adj = torch.rand((10, 10))

    mock = MagicMock()
    monkeypatch.setattr("src.data.utils.remove_edges_from_undirected_graph", mock)

    remove_edges(adj, is_directed=False, remove_edges_percentage=0.5, seed=42)
    mock.assert_called_once()


def test_remove_edges_from_directed_graph_returns_correct_result():
    m = np.eye(10)
    m[2] = 1.0

    dense_adj = torch.as_tensor(m).float()
    dense_adj_removed = remove_edges_from_directed_graph(dense_adj, 1.0)

    assert (dense_adj_removed.sum() == 0.0) and \
           (torch.equal(dense_adj_removed, torch.zeros(dense_adj.size())))

    dense_adj_removed = remove_edges_from_directed_graph(dense_adj, 0.0)
    assert torch.equal(dense_adj_removed, dense_adj)

    dense_adj = torch.as_tensor(np.eye(2)).float()
    dense_adj_removed = remove_edges_from_directed_graph(dense_adj, 0.5)
    assert dense_adj_removed.sum() == 1.0


def test_remove_edges_from_directed_graph_returns_expected_graph():
    adj = torch.as_tensor([
        [0.0, 0.0, 0.5],
        [0.1, 0.0, 0.4],
        [0.0, 0.0, 0.2],
    ])
    allowed_results = [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.2],
        ],
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
        ],
        [
            [0.0, 0.0, 0.5],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.4],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0],
        ]
    ]
    for seed in range(0, 100):
        result = remove_edges_from_directed_graph(adj=adj, remove_edges_percentage=0.5, seed=seed)
        is_in_allowed_results = [torch.allclose(result, torch.as_tensor(res)) for res in allowed_results]
        assert sum(is_in_allowed_results) == 1


def test_remove_edges_from_directed_graph_returns_unweighted_graph_if_one_is_given():
    adj = torch.rand((20, 20)).bernoulli()
    removed = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5)
    assert set(removed.flatten().tolist()).issubset([0.0, 1.0])


def test_remove_edges_from_directed_graph_returns_weighted_graph_if_one_is_given():
    adj = torch.rand((20, 20))
    removed = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5)
    assert set(removed.flatten().tolist()).issubset(adj.flatten().tolist() + [0.0])


def test_remove_edges_from_directed_graph_does_not_remove_any_if_percentage_is_zero():
    adj = torch.rand((20, 20)).bernoulli()
    removed = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.0)
    assert removed.equal(adj)


def test_remove_edges_from_directed_graph_removes_all_if_percentage_is_one():
    adj = torch.rand((20, 20)).bernoulli()
    removed = remove_edges_from_directed_graph(adj, remove_edges_percentage=1.0)
    assert removed.equal(torch.zeros_like(adj))


def test_remove_edges_from_directed_graph_returns_same_graphs_for_same_seeds():
    adj = torch.rand((20, 20)).bernoulli()
    removed1 = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5, seed=42)
    removed2 = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5, seed=42)
    assert removed1.equal(removed2)


def test_remove_edges_from_directed_graph_returns_different_graphs_for_different_seeds():
    adj = torch.rand((20, 20)).bernoulli()
    removed1 = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5, seed=None)
    removed3 = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5, seed=42)
    removed4 = remove_edges_from_directed_graph(adj, remove_edges_percentage=0.5, seed=33)
    assert not removed1.equal(removed3)
    assert not removed3.equal(removed4)


def test_remove_edges_from_undirected_weighted_graph_returns_weighted_graph():
    adj = torch.rand((20, 20))
    adj = to_undirected(adj, from_triu_only=False)
    removed = remove_edges_from_undirected_graph(adj, remove_edges_percentage=0.5)
    assert not removed.flatten().unique(sorted=True).equal(torch.as_tensor([0.0, 1.0]))


def test_remove_edges_from_undirected_unweighted_fails_for_directed_graph():
    adj = torch.rand((20, 20)).bernoulli()
    with pytest.raises(AssertionError):
        remove_edges_from_undirected_graph(adj, remove_edges_percentage=0.5)


def test_remove_edges_from_undirected_unweighted_graph_returns_unweighted_graph():
    adj = torch.rand((20, 20)).bernoulli()
    adj = to_undirected(adj, from_triu_only=False)
    removed = remove_edges_from_undirected_graph(adj, remove_edges_percentage=0.5)
    assert removed.flatten().unique(sorted=True).equal(torch.as_tensor([0.0, 1.0]))


@pytest.mark.parametrize("remove_edges_percentage", [0.0, 0.25, 0.75, 1.0])
def test_remove_edges_from_undirected_graph_returns_undirected_graph(remove_edges_percentage):
    adj = torch.rand((20, 20)).bernoulli()
    adj = to_undirected(adj, from_triu_only=False)
    removed = remove_edges_from_undirected_graph(adj, remove_edges_percentage=remove_edges_percentage)
    assert get_triu_values(removed).equal(get_triu_values(removed.t()))


def test_remove_edges_from_undirected_graph_does_not_remove_any_if_percentage_is_zero():
    adj = torch.rand((20, 20)).bernoulli()
    adj = to_undirected(adj, from_triu_only=False)
    removed = remove_edges_from_undirected_graph(adj, remove_edges_percentage=0.0)
    assert adj.equal(removed)


def test_load_function_normalizes_features(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data = load_process_dataset(dataset="cora",
                                    normalize_features=True,
                                    remove_edges_percentage=.5,
                                    shuffle_splits=True,
                                    make_undirected=True,
                                    nearest_neighbor_k=10,
                                    use_largest_subgraph=True
                                    )

        # Remove Zero Rows
        data.x = data.x[data.x.sum(dim=1).nonzero(), :].squeeze()
        assert data.x.sum(dim=1).allclose(torch.as_tensor(1.0))


def test_load_function_does_not_normalize_features(cora):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data = load_process_dataset(dataset="cora",
                                    normalize_features=False,
                                    remove_edges_percentage=.5,
                                    shuffle_splits=True,
                                    make_undirected=True,
                                    nearest_neighbor_k=10,
                                    use_largest_subgraph=True
                                    )

        # Remove Zero Rows
        data.x = data.x[data.x.sum(dim=1).nonzero(), :].squeeze()
        assert not data.x.sum(dim=1).allclose(torch.as_tensor(1.0))


@pytest.mark.parametrize(
    argnames=["dataset", "remove_edges_percentage", "normalize_features", "shuffle_splits", "make_undirected",
              "nearest_neighbor_k", "use_largest_subgraph"],
    argvalues=product(GRAPH_DATASETS, [0.0, 0.5], [True, False], [True, False], [True, False], [None, 10],
                      [True, False]))
def test_load_process_graph_dataset_runs_without_error(dataset: str,
                                                       remove_edges_percentage: float,
                                                       normalize_features: bool,
                                                       shuffle_splits: bool,
                                                       make_undirected: bool,
                                                       nearest_neighbor_k: int,
                                                       use_largest_subgraph: bool,
                                                       monkeypatch,
                                                       citeseer):
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=cora) as _:
        data = load_process_dataset(dataset=dataset,
                                    remove_edges_percentage=remove_edges_percentage,
                                    normalize_features=normalize_features,
                                    shuffle_splits=shuffle_splits,
                                    make_undirected=make_undirected,
                                    nearest_neighbor_k=nearest_neighbor_k,
                                    use_largest_subgraph=use_largest_subgraph
                                    )


@pytest.mark.parametrize(argnames="dataset", argvalues=UCI_DATASETS)
def test_load_process_uci_dataset_fails_without_random_splits(dataset: str,
                                                              monkeypatch):
    with pytest.raises(AssertionError):
        data = load_process_dataset(dataset=dataset,
                                    remove_edges_percentage=0.3,
                                    normalize_features=True,
                                    shuffle_splits=False,
                                    make_undirected=True,
                                    nearest_neighbor_k=10,
                                    use_largest_subgraph=True
                                    )


@pytest.mark.parametrize(
    argnames=["dataset", "remove_edges_percentage", "normalize_features", "make_undirected", "nearest_neighbor_k"],
    argvalues=product(UCI_DATASETS, [0.0, 0.5], [True, False], [True, False], [None, 10]))
def test_load_process_uci_dataset_runs_without_errors(dataset: str,
                                                      remove_edges_percentage: float,
                                                      normalize_features: bool,
                                                      make_undirected: bool,
                                                      nearest_neighbor_k: int,
                                                      monkeypatch
                                                      ):
    data = load_process_dataset(dataset=dataset,
                                remove_edges_percentage=remove_edges_percentage,
                                normalize_features=normalize_features,
                                shuffle_splits=True,
                                make_undirected=make_undirected,
                                nearest_neighbor_k=nearest_neighbor_k,
                                use_largest_subgraph=False
                                )


@pytest.mark.parametrize(
    argnames=["dataset", "remove_edges_percentage", "normalize_features", "shuffle_splits", "make_undirected",
              "nearest_neighbor_k", "use_largest_subgraph"],
    argvalues=product(GRAPH_DATASETS, [0.0, 0.5], [True, False], [True, False], [True, False], [None, 10],
                      [True, False]))
def test_load_process_graph_dataset_runs_without_error(dataset: str,
                                                       remove_edges_percentage: float,
                                                       normalize_features: bool,
                                                       shuffle_splits: bool,
                                                       make_undirected: bool,
                                                       nearest_neighbor_k: int,
                                                       use_largest_subgraph: bool,
                                                       monkeypatch):
    dataset_load_function = cora if dataset == "cora" else citeseer
    with mock.patch("src.data.utils.load_planetoid_dataset", return_value=dataset_load_function) as _:
        data = load_process_dataset(dataset=dataset,
                                    remove_edges_percentage=remove_edges_percentage,
                                    normalize_features=normalize_features,
                                    shuffle_splits=shuffle_splits,
                                    make_undirected=make_undirected,
                                    nearest_neighbor_k=nearest_neighbor_k,
                                    use_largest_subgraph=use_largest_subgraph
                                    )
