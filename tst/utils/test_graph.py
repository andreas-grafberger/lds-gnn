from itertools import product

import pytest

import torch
from torch_geometric.datasets import Planetoid

from src.utils.graph import to_undirected, get_triu_values, split_mask, to_dense_adj, is_square_matrix, add_self_loops, \
    normalize_adjacency_matrix, cosine_similarity, triu_values_to_symmetric_matrix, num_nodes_from_triu_shape, \
    dirichlet_energy, disconnection_loss, sparsity_loss, graph_regularization
from tst.test_utils import resource_folder_path

N_NODES = 10


@pytest.fixture(scope="package")
def cora(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    return Planetoid(tmpdir, "cora")


@pytest.fixture
def small_mask():
    return torch.as_tensor([True, False, False, True, False, True])


@pytest.fixture
def data_mask():
    return torch.bernoulli(torch.as_tensor(0.5).expand(100000)).bool()


def test_to_undirected_from_directed_uses_full_matrix():
    matrix = torch.zeros((N_NODES, N_NODES))
    matrix[:, 1] = 1.0
    undirected_computed = to_undirected(matrix, from_triu_only=False)

    expected = torch.zeros_like(matrix)
    expected[:, 1] = 1.0
    expected[1, :] = 1.0
    assert expected.allclose(undirected_computed)


def test_to_undirected_from_directed_uses_triu_matrix():
    matrix = torch.zeros((N_NODES, N_NODES))
    matrix[:, 1] = 1.0
    undirected_computed = to_undirected(matrix, from_triu_only=True)

    expected = torch.zeros_like(matrix)
    expected[0, 1] = 1.0
    expected[1, 1] = 1.0
    expected[1, 0] = 1.0
    assert expected.allclose(undirected_computed)


def test_to_undirected_grad_flow_in_triu_and_tril():
    matrix = (torch.rand((N_NODES, N_NODES)) * 0.5).requires_grad_(True)
    assert matrix.grad is None
    undirected_computed = to_undirected(matrix, from_triu_only=False)
    undirected_computed.sum().backward()
    assert matrix.grad is not None
    assert matrix.grad.triu(diagonal=1).sum() > 0
    assert matrix.grad.tril(diagonal=-1).sum() > 0
    # assert matrix.grad.nonzero().size(0) == matrix.numel()


def test_to_undirected_grad_flows_into_triu_only():
    matrix = (torch.rand((5, 5)) * 0.5).requires_grad_(True)
    assert matrix.grad is None
    undirected_computed = to_undirected(matrix, from_triu_only=True)
    undirected_computed.sum().backward()
    assert matrix.grad is not None
    assert matrix.grad.nonzero().size(0) == 15


def test_get_triu_values_preserves_gradient_flow():
    matrix = torch.rand((5, 5)).requires_grad_(True)
    assert matrix.grad is None

    triu_values = get_triu_values(matrix)
    triu_values.sum().backward()

    assert matrix.grad is not None
    assert matrix.grad.nonzero().size(0) == 15


def test_split_masks_do_not_overlap_and_are_correct(small_mask):
    first_mask, second_mask = split_mask(mask=small_mask,
                                         ratio=0.7,
                                         shuffle=False)
    assert first_mask.nonzero(as_tuple=False).squeeze().equal(torch.as_tensor([0, 3]))
    assert second_mask.nonzero(as_tuple=False).equal(torch.as_tensor([[5]]))
    assert (first_mask | second_mask).equal(small_mask)
    assert (first_mask ^ second_mask).equal(small_mask)


@pytest.mark.parametrize(["ratio"], product([0.0, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0]))
def test_split_masks_do_not_overlap_for_different_rations(data_mask, ratio):
    first_mask, second_mask = split_mask(mask=data_mask,
                                         ratio=ratio,
                                         shuffle=False)
    assert (first_mask | second_mask).equal(data_mask)
    assert (first_mask ^ second_mask).equal(data_mask)


@pytest.mark.parametrize(["seed"], product([42, 1231231, 2412312312, 64365435]))
def test_split_masks_do_not_overlap_for_different_shuffles(data_mask, seed):
    torch.manual_seed(seed)
    first_mask, second_mask = split_mask(mask=data_mask,
                                         ratio=0.5,
                                         shuffle=True)
    assert (first_mask | second_mask).equal(data_mask)
    assert (first_mask ^ second_mask).equal(data_mask)


def test_to_dense_adj_creates_correct_matrix_without_num_nodes_specified():
    edge_index = torch.as_tensor([[0, 1, 2, 3, 4, 0, 1], [0, 1, 2, 3, 4, 1, 5]])
    calculated = to_dense_adj(edge_index)
    expected = torch.eye(6)
    expected[5, 5] = 0
    expected[0, 1] = 1
    expected[1, 5] = 1

    assert expected.equal(calculated)


def test_to_dense_adj_creates_correct_matrix_starting_from_nonzero():
    edge_index = torch.as_tensor([[1, 2, 3, 4, 1], [1, 2, 3, 4, 5]])
    calculated = to_dense_adj(edge_index)
    expected = torch.eye(6)
    expected[5, 5] = 0
    expected[0, 0] = 0
    expected[1, 5] = 1

    assert expected.equal(calculated)


def test_to_dense_adj_creates_correct_matrix_with_num_max_nodes_specified():
    edge_index = torch.as_tensor([[1, 2, 3, 4, 1], [1, 2, 3, 4, 5]])
    calculated = to_dense_adj(edge_index, num_max_nodes=100)
    expected = torch.zeros(100, 100)
    expected[torch.eye(6).nonzero(as_tuple=True)] = 1
    expected[5, 5] = 0
    expected[0, 0] = 0
    expected[1, 5] = 1

    assert expected.equal(calculated)


def test_is_square_matrix():
    matrix = torch.rand((5, 5, 4))
    assert not is_square_matrix(matrix)

    matrix = torch.rand((5, 5, 1))
    assert not is_square_matrix(matrix)

    matrix = torch.rand((4, 5))
    assert not is_square_matrix(matrix)

    matrix = torch.rand((5, 4))
    assert not is_square_matrix(matrix)

    matrix = torch.rand((5))
    assert not is_square_matrix(matrix)

    matrix = torch.rand((5, 5))
    assert is_square_matrix(matrix)


def test_add_self_loops_preserves_non_diagonal_gradients():
    matrix = torch.rand((100, 100)).requires_grad_(True)
    calculated = add_self_loops(matrix)
    calculated.sum().backward()
    assert matrix.grad is not None
    expected_nonzero_grad_indices = torch.cat((torch.triu_indices(100, 100, 1),
                                               torch.tril_indices(100, 100, -1)), dim=1)
    expected_nonzero_grad_indices = torch.sort(expected_nonzero_grad_indices)[0]
    calculated_nonzero_grad_indices = torch.sort(matrix.grad.nonzero().t())[0]
    assert calculated_nonzero_grad_indices.equal(expected_nonzero_grad_indices)


def test_adj_normalized_just_like_in_other_repo(cora):
    """
    Compares our adjacency normalization function with the ones in the repo https://github.com/dragen1860/GCN-PyTorch.
    The file was generated by running their code and saving the dense matrix via torch.save
    """
    # Load Reference Adjacency Matrix
    path = resource_folder_path() / "gcn_pytorch_normalized_adj.pt"
    with path.open("rb") as f:
        reference_adj = torch.load(f)

    # Preprocess with our code
    data = cora[0]
    dense_adj = to_dense_adj(data.edge_index, num_max_nodes=data.num_nodes)
    normalized_adj = normalize_adjacency_matrix(dense_adj)
    assert torch.allclose(normalized_adj, reference_adj)


def test_cosine_similarity_in_range():
    matrix = torch.rand(500, 1024)
    sim_matrix = cosine_similarity(matrix)
    print(sim_matrix.max())
    assert (sim_matrix >= -1.0).all()
    assert (sim_matrix <= 1.0).all()


def test_cosine_similarity_preserves_gradient_flow():
    matrix = torch.rand(512, 512).requires_grad_(True)
    sim_matrix = cosine_similarity(matrix)
    sim_matrix.sum().backward()
    assert (matrix.grad != 0.0).all()


def test_triu_values_to_symmetric_matrix():
    triu_values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    expected_matrix = torch.as_tensor([
        [0.1, 0.2, 0.3],
        [0.2, 0.4, 0.5],
        [0.3, 0.5, 0.6]
    ])
    calculated_matrix = triu_values_to_symmetric_matrix(triu_values)
    assert expected_matrix.equal(calculated_matrix)


def test_triu_values_to_symmetric_matrix_preserves_gradient_flow():
    triu_values = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).requires_grad_(True)
    calculated_matrix = triu_values_to_symmetric_matrix(triu_values)
    calculated_matrix.sum().backward()
    assert triu_values.grad is not None
    assert (triu_values.grad != 0.0).all()


@pytest.mark.parametrize("nodes", [10, 100, 1000, 2000, 500000])
def test_num_nodes_from_triu_shape(nodes):
    n_triu_values = ((nodes ** 2) / 2) + nodes / 2
    assert nodes == num_nodes_from_triu_shape(n_triu_values)


def test_dirichlet_energy_positive():
    adj = torch.bernoulli(torch.rand(500, 500)).float()
    features = torch.rand(500, 128) * 1000
    loss = dirichlet_energy(adj, features)
    assert 0.0 <= loss


def test_dirichlet_energy_differentiable():
    adj = torch.rand((500, 500)).requires_grad_(True)
    features = (torch.rand((500, 128)) * 1000).requires_grad_(True)
    loss = dirichlet_energy(adj, features)
    loss.backward()
    assert adj.grad is not None
    assert ((adj.grad + torch.eye(500)) != 0.0).all()
    assert features.grad is not None
    assert (features.grad != 0.0).all()


def test_disconnection_loss_differentiable():
    adj = torch.rand((500, 500)).requires_grad_(True)
    loss = disconnection_loss(adj)
    loss.backward()
    assert adj.grad is not None
    assert ((adj.grad + torch.eye(500)) != 0.0).all()


def test_sparsity_loss_in_valid_range():
    for seed in range(0, 111):
        torch.manual_seed(seed)
        adj = torch.rand((500, 500))
        loss = sparsity_loss(adj)
        assert loss >= 0.0


def test_sparsity_loss_differentiable():
    adj = torch.rand((500, 500)).requires_grad_(True)
    loss = sparsity_loss(adj)
    loss.backward()
    assert adj.grad is not None
    assert ((adj.grad + torch.eye(500)) != 0.0).all()


def test_graph_regularization_always_positive():
    for seed in range(0, 111):
        torch.manual_seed(seed)
        adj = torch.rand((500, 500))
        features = (torch.rand((500, 128)) * 1000)
        loss = graph_regularization(adj, features,
                                    smoothness_factor=1.0,
                                    disconnection_factor=1.0,
                                    sparsity_factor=1.0,
                                    log=False)
        assert loss >= 0.0


def test_graph_regularization_always_differentiable():
    for seed in range(0, 111):
        torch.manual_seed(seed)
        adj = torch.rand((500, 500)).requires_grad_(True)
        features = (torch.rand((500, 128)) * 1000).requires_grad_(True)
        loss = graph_regularization(adj, features,
                                    smoothness_factor=1.0,
                                    disconnection_factor=1.0,
                                    sparsity_factor=1.0,
                                    log=False)
        loss.backward()
        assert adj.grad is not None
        assert ((adj.grad + torch.eye(500)) != 0.0).all()
        assert features.grad is not None
        assert (features.grad != 0.0).all()
