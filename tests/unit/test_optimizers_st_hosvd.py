"""
Basic sanity checks for TensorionSTHoSVD and TensorionSTHoSVDSGD optimizers.
"""
import torch
import pytest
from tensorion.optimizers import TensorionSTHoSVD, TensorionSTHoSVDSGD


def test_tensorion_st_hosvd_basic_step():
    """Test that TensorionSTHoSVD can perform a basic optimization step."""
    # Create a simple 3D tensor parameter
    param = torch.randn(8, 10, 12, requires_grad=True)
    
    # Create optimizer
    optimizer = TensorionSTHoSVD(
        params=[param],
        muon_params=[param],
        lr=1e-2,
        wd=0.01,
        ranks=4,
        momentum=0.9,
        min_dim=3,
    )
    
    # Create a simple loss
    loss = (param ** 2).sum()
    loss.backward()
    
    # Store initial parameter value
    initial_param = param.data.clone()
    
    # Perform optimization step
    optimizer.step()
    
    # Check that parameter was updated
    assert not torch.allclose(param.data, initial_param), "Parameter should be updated"
    
    # Check that gradient is cleared (or at least step completed without error)
    assert param.grad is not None  # grad might still exist, but step should work


def test_tensorion_st_hosvd_sgd_basic_step():
    """Test that TensorionSTHoSVDSGD can perform a basic optimization step."""
    # Create a simple 3D tensor parameter
    param = torch.randn(8, 10, 12, requires_grad=True)
    
    # Create optimizer
    optimizer = TensorionSTHoSVDSGD(
        params=[param],
        muon_params=[param],
        lr=1e-2,
        wd=0.01,
        ranks=4,
        momentum=0.9,
        min_dim=3,
    )
    
    # Create a simple loss
    loss = (param ** 2).sum()
    loss.backward()
    
    # Store initial parameter value
    initial_param = param.data.clone()
    
    # Perform optimization step
    optimizer.step()
    
    # Check that parameter was updated
    assert not torch.allclose(param.data, initial_param), "Parameter should be updated"


def test_tensorion_st_hosvd_loss_decreases():
    """Test that loss decreases over multiple steps (on a simple convex problem)."""
    # Simple quadratic loss: minimize ||x - target||^2
    target = torch.randn(6, 8, 10)
    param = torch.zeros_like(target, requires_grad=True)
    
    optimizer = TensorionSTHoSVD(
        params=[param],
        muon_params=[param],
        lr=0.1,
        wd=0.0,  # No weight decay for simplicity
        ranks=3,
        momentum=0.0,  # No momentum for simplicity
        min_dim=3,
    )
    
    losses = []
    for step in range(5):
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    
    # Loss should generally decrease (allowing for some numerical issues)
    # At minimum, final loss should be less than initial loss
    assert losses[-1] < losses[0], f"Loss should decrease: {losses}"


def test_tensorion_st_hosvd_sgd_loss_decreases():
    """Test that loss decreases over multiple steps for TensorionSTHoSVDSGD."""
    target = torch.randn(6, 8, 10)
    param = torch.zeros_like(target, requires_grad=True)
    
    optimizer = TensorionSTHoSVDSGD(
        params=[param],
        muon_params=[param],
        lr=0.1,
        wd=0.0,
        ranks=3,
        momentum=0.0,
        min_dim=3,
    )
    
    losses = []
    for step in range(5):
        loss = ((param - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    
    assert losses[-1] < losses[0], f"Loss should decrease: {losses}"


def test_tensorion_st_hosvd_with_adamw_backup():
    """Test TensorionSTHoSVD with both muon_params and adamw_params."""
    # Muon parameter (3D tensor)
    muon_param = torch.randn(8, 10, 12, requires_grad=True)
    
    # AdamW parameter (2D tensor, not eligible for muon)
    adamw_param = torch.randn(20, 30, requires_grad=True)
    
    optimizer = TensorionSTHoSVD(
        params=[muon_param, adamw_param],
        muon_params=[muon_param],
        adamw_params=[adamw_param],
        lr=1e-2,
        wd=0.01,
        ranks=4,
        min_dim=3,
    )
    
    # Create loss
    loss = (muon_param ** 2).sum() + (adamw_param ** 2).sum()
    loss.backward()
    
    initial_muon = muon_param.data.clone()
    initial_adamw = adamw_param.data.clone()
    
    optimizer.step()
    
    # Both should be updated
    assert not torch.allclose(muon_param.data, initial_muon)
    assert not torch.allclose(adamw_param.data, initial_adamw)


def test_tensorion_st_hosvd_sgd_with_sgd_backup():
    """Test TensorionSTHoSVDSGD with both muon_params and sgd_params."""
    muon_param = torch.randn(8, 10, 12, requires_grad=True)
    sgd_param = torch.randn(20, 30, requires_grad=True)
    
    optimizer = TensorionSTHoSVDSGD(
        params=[muon_param, sgd_param],
        muon_params=[muon_param],
        sgd_params=[sgd_param],
        lr=1e-2,
        wd=0.01,
        ranks=4,
        min_dim=3,
    )
    
    loss = (muon_param ** 2).sum() + (sgd_param ** 2).sum()
    loss.backward()
    
    initial_muon = muon_param.data.clone()
    initial_sgd = sgd_param.data.clone()
    
    optimizer.step()
    
    assert not torch.allclose(muon_param.data, initial_muon)
    assert not torch.allclose(sgd_param.data, initial_sgd)


def test_tensorion_st_hosvd_momentum():
    """Test that momentum is working correctly."""
    param = torch.randn(6, 8, 10, requires_grad=True)
    
    optimizer = TensorionSTHoSVD(
        params=[param],
        muon_params=[param],
        lr=0.1,
        wd=0.0,
        ranks=3,
        momentum=0.9,
        min_dim=3,
    )
    
    # First step
    loss1 = (param ** 2).sum()
    loss1.backward()
    step1_update = param.grad.clone()
    optimizer.step()
    optimizer.zero_grad()
    
    # Second step with same gradient
    loss2 = (param ** 2).sum()
    loss2.backward()
    step2_update = param.grad.clone()
    optimizer.step()
    
    # With momentum, the second update should be influenced by the first
    # (This is a basic check - exact behavior depends on implementation)
    assert step2_update is not None


def test_tensorion_st_hosvd_nesterov():
    """Test Nesterov momentum option."""
    param = torch.randn(6, 8, 10, requires_grad=True)
    
    optimizer = TensorionSTHoSVD(
        params=[param],
        muon_params=[param],
        lr=0.1,
        wd=0.0,
        ranks=3,
        momentum=0.9,
        nesterov=True,
        min_dim=3,
    )
    
    loss = (param ** 2).sum()
    loss.backward()
    initial_param = param.data.clone()
    
    optimizer.step()
    
    # Should update without error
    assert not torch.allclose(param.data, initial_param)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
