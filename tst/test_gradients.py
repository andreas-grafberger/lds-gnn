import pytest
import torch

from src.utils.graph import add_self_loops


def test_gradient_flow_does_not_work_with_inplace_operation():
    params = torch.ones((5, 5), requires_grad=True)
    with pytest.raises(RuntimeError):
        params.fill_diagonal_(1.0)
        x = params.sum()
        x.backward()


def test_gradient_flow_does_not_work_with_clone_and_inplace_operation():
    params = torch.ones((5, 5), requires_grad=True)
    params_clone = params.clone()
    params_clone.fill_diagonal_(1.0)
    x = params_clone.sum()
    x.backward()
    assert params.grad is not None


def test_gradient_flow_does_work_with_clamp_():
    params = torch.ones((5, 5), requires_grad=True)
    params_self_loop = params + torch.eye(5)
    params_self_loop.clamp_(0.0, 1.0)

    assert params.grad is None
    x = params_self_loop.sum()
    x.backward()
    assert params.grad is not None


def test_gradient_flow_does_work_with_clamp():
    params_orig = torch.ones((5, 5), requires_grad=True)
    params = params_orig + torch.eye(5)
    params = params.clamp(0.0, 1.0)

    assert params_orig.grad is None
    x = params.sum()
    x.backward()
    assert params_orig.grad is not None


def test_diag_preserves_gradient_flow():
    params = torch.ones((5,), requires_grad=True)
    x = torch.diag(params)
    x.sum().backward()
    assert params.grad is not None


def test_add_self_loops_preserves_gradient_flow():
    adj = torch.eye(5).requires_grad_(True)
    p = add_self_loops(adj)
    p.sum().backward()
    assert adj.grad is not None
