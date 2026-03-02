"""
Test the improved parameter handling logic in optimizers.
"""
import torch
import pytest
from tensorion.optimizers import TensorionSGD, STHoSVDTensorion
from tensorion.optimizers import factory as optim_factory


def test_params_as_base_set():
    """Test that params is the base set and muon_params/sgd_params are subsets."""
    # Create parameters
    muon_param = torch.randn(8, 10, 12, requires_grad=True)
    sgd_param = torch.randn(20, 30, requires_grad=True)
    other_param = torch.randn(5, 6, 7, requires_grad=True)
    
    # All params together
    all_params = [muon_param, sgd_param, other_param]
    
    # Create optimizer with params as base, muon_params and sgd_params as subsets
    optimizer = TensorionSGD(
        params=all_params,
        muon_params=[muon_param, other_param],  # Subset of params
        sgd_params=[sgd_param],  # Subset of params
        lr=1e-2,
        min_dim=3,
    )
    
    # Verify that muon_params are marked correctly
    assert optimizer.state[muon_param]["use_muon"] == True
    assert optimizer.state[other_param]["use_muon"] == True
    assert optimizer.state[sgd_param]["use_muon"] == False
    
    # Test that optimization step works
    loss = (muon_param ** 2).sum() + (sgd_param ** 2).sum() + (other_param ** 2).sum()
    loss.backward()
    optimizer.step()
    
    # Verify parameters were updated
    assert muon_param.grad is not None
    assert sgd_param.grad is not None
    assert other_param.grad is not None


def test_auto_detection_when_muon_params_none():
    """Test auto-detection when muon_params is None."""
    # Create parameters with different dimensions
    high_dim_param = torch.randn(8, 10, 12, requires_grad=True)  # 3D
    low_dim_param = torch.randn(20, 30, requires_grad=True)  # 2D
    
    all_params = [high_dim_param, low_dim_param]
    
    # Create optimizer without specifying muon_params (should auto-detect)
    optimizer = TensorionSGD(
        params=all_params,
        muon_params=None,  # Auto-detect
        sgd_params=None,
        lr=1e-2,
        min_dim=3,
    )
    
    # High-dim param should get muon, low-dim should not
    assert optimizer.state[high_dim_param]["use_muon"] == True
    assert optimizer.state[low_dim_param]["use_muon"] == False


def test_error_on_invalid_subset():
    """Test that error is raised if muon_params/sgd_params contain params not in params."""
    param1 = torch.randn(8, 10, 12, requires_grad=True)
    param2 = torch.randn(20, 30, requires_grad=True)
    param3 = torch.randn(5, 6, 7, requires_grad=True)  # Not in params
    
    # param3 is not in params, so should raise error
    with pytest.raises(ValueError, match="muon_params contains parameters not in params"):
        TensorionSGD(
            params=[param1, param2],
            muon_params=[param1, param3],  # param3 not in params
            sgd_params=None,
            lr=1e-2,
            min_dim=3,
        )


def test_error_on_overlap():
    """Test that error is raised if param is in both muon_params and sgd_params."""
    param1 = torch.randn(8, 10, 12, requires_grad=True)
    param2 = torch.randn(20, 30, requires_grad=True)
    
    with pytest.raises(ValueError, match="Parameters cannot be in both"):
        TensorionSGD(
            params=[param1, param2],
            muon_params=[param1],
            sgd_params=[param1],  # Same param in both
            lr=1e-2,
            min_dim=3,
        )


def test_st_hosvd_with_new_logic():
    """Test STHoSVDTensorion with the new parameter logic."""
    muon_param = torch.randn(8, 10, 12, requires_grad=True)
    adamw_param = torch.randn(20, 30, requires_grad=True)
    other_param = torch.randn(5, 6, 7, requires_grad=True)
    
    all_params = [muon_param, adamw_param, other_param]
    
    optimizer = STHoSVDTensorion(
        params=all_params,
        muon_params=[muon_param, other_param],
        adamw_params=[adamw_param],
        lr=1e-2,
        ranks=3,
        min_dim=3,
    )
    
    assert optimizer.state[muon_param]["use_muon"] == True
    assert optimizer.state[other_param]["use_muon"] == True
    assert optimizer.state[adamw_param]["use_muon"] == False
    
    # Test optimization step
    loss = (muon_param ** 2).sum() + (adamw_param ** 2).sum() + (other_param ** 2).sum()
    loss.backward()
    optimizer.step()
    
    # All should be updated
    assert muon_param.grad is not None
    assert adamw_param.grad is not None
    assert other_param.grad is not None


class _SimpleModel(torch.nn.Module):
    """Small model to expose a realistic parameter iterator (model.parameters())."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3)
        self.fc = torch.nn.Linear(4 * 30 * 30, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_tensorion_sgd_factory_accepts_parameter_iterator():
    """
    Regression test for iterator exhaustion in create_tensorion_sgd_optimizer.

    The factory used to iterate over a one-shot iterable (model.parameters())
    to build muon/sgd parameter lists and then pass the already-consumed
    iterator into TensorionSGD, leading to an empty params list and a
    ValueError from the subset validation.
    """
    model = _SimpleModel()

    # Use the real parameter iterator, not a pre-built list
    optimizer = optim_factory.create_tensorion_sgd_optimizer(
        params=model.parameters(),
        lr=1e-2,
        min_dim=3,
    )

    x = torch.randn(2, 3, 32, 32)
    out = model(x).sum()
    out.backward()

    # Should run without raising and update parameters
    optimizer.step()


def test_st_hosvd_tensorion_sgd_factory_accepts_parameter_iterator():
    """Same iterator-exhaustion regression test for the ST-HOSVD TensorionSGD variant."""
    model = _SimpleModel()

    optimizer = optim_factory.create_st_hosvd_tensorion_sgd_optimizer(
        params=model.parameters(),
        lr=1e-2,
        ranks=2,
        min_dim=3,
    )

    x = torch.randn(2, 3, 32, 32)
    out = model(x).sum()
    out.backward()

    optimizer.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
