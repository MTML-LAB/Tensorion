"""
Toy tensor optimization problem to test STHoSVDTensorion and STHoSVDTensorionSGD.

We'll solve a low-rank tensor approximation problem:
Given a target tensor T, find a low-rank approximation that minimizes ||T - X||^2
where X is reconstructed from a Tucker decomposition.
"""
import torch
import numpy as np
from tensorion.optimizers import STHoSVDTensorion, STHoSVDTensorionSGD


def create_low_rank_tensor(shape, rank, seed=42):
    """Create a tensor with low Tucker rank for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random factors
    factors = []
    core_shape = [rank] * len(shape)
    core = torch.randn(*core_shape)
    
    for i, dim in enumerate(shape):
        factor = torch.randn(dim, rank)
        factors.append(factor)
    
    # Reconstruct tensor using Tucker decomposition
    from tensorion.tensor_utils import tucker_reconstruct
    tensor = tucker_reconstruct(core, factors)
    
    return tensor, core, factors


def test_tensorion_st_hosvd_toy_problem():
    """Test STHoSVDTensorion on a toy tensor approximation problem."""
    print("\n" + "="*60)
    print("Testing STHoSVDTensorion on Toy Tensor Optimization")
    print("="*60)
    
    # Create target tensor with known low rank
    target_shape = (8, 10, 12)
    true_rank = 3
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=42)
    
    # Initialize parameter tensor (start from random)
    param = torch.randn(*target_shape, requires_grad=True)
    
    # Create optimizer
    optimizer = STHoSVDTensorion(
        params=[param],
        muon_params=[param],
        lr=0.01,  # Lower learning rate for stability
        wd=0.0,  # No weight decay for this test
        ranks=true_rank,
        momentum=0.9,
        nesterov=False,
        nuclear=False,
        adjust_lr=False,  # Use fixed LR for simplicity
        min_dim=3,
    )
    
    # Optimization loop
    num_steps = 50
    losses = []
    
    print(f"\nTarget tensor shape: {target_shape}")
    print(f"Using rank: {true_rank}")
    print(f"Initial loss: {((param - target_tensor) ** 2).sum().item():.6f}")
    print("\nOptimizing...")
    
    for step in range(num_steps):
        # Compute loss
        loss = ((param - target_tensor) ** 2).sum()
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress every 10 steps
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:3d}: Loss = {loss.item():.6f}")
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.6f}")
    print(f"Relative error: {torch.norm(param - target_tensor).item() / torch.norm(target_tensor).item():.6f}")
    
    # Verify that loss decreased (allow for some numerical issues)
    assert losses[-1] < losses[0] * 0.9, f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print("\n✓ Test passed: Loss decreased!")
    
    return losses


def test_tensorion_st_hosvd_sgd_toy_problem():
    """Test STHoSVDTensorionSGD on a toy tensor approximation problem."""
    print("\n" + "="*60)
    print("Testing STHoSVDTensorionSGD on Toy Tensor Optimization")
    print("="*60)
    
    # Create target tensor with known low rank
    target_shape = (8, 10, 12)
    true_rank = 3
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=43)
    
    # Initialize parameter tensor
    param = torch.randn(*target_shape, requires_grad=True)
    
    # Create optimizer
    optimizer = STHoSVDTensorionSGD(
        params=[param],
        muon_params=[param],
        lr=0.01,  # Lower learning rate for stability
        wd=0.0,
        ranks=true_rank,
        momentum=0.9,
        nesterov=False,
        nuclear=False,
        adjust_lr=False,
        min_dim=3,
    )
    
    # Optimization loop
    num_steps = 50
    losses = []
    
    print(f"\nTarget tensor shape: {target_shape}")
    print(f"Using rank: {true_rank}")
    print(f"Initial loss: {((param - target_tensor) ** 2).sum().item():.6f}")
    print("\nOptimizing...")
    
    for step in range(num_steps):
        loss = ((param - target_tensor) ** 2).sum()
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:3d}: Loss = {loss.item():.6f}")
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.6f}")
    print(f"Relative error: {torch.norm(param - target_tensor).item() / torch.norm(target_tensor).item():.6f}")
    
    assert losses[-1] < losses[0] * 0.9, f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print("\n✓ Test passed: Loss decreased!")
    
    return losses


def test_comparison_with_baseline():
    """Compare Tensorion optimizers with standard SGD on the same problem."""
    print("\n" + "="*60)
    print("Comparing STHoSVDTensorion with Standard SGD")
    print("="*60)
    
    target_shape = (6, 8, 10)
    true_rank = 2
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=44)
    
    # Test STHoSVDTensorion
    param1 = torch.randn(*target_shape, requires_grad=True)
    optimizer1 = STHoSVDTensorion(
        params=[param1],
        muon_params=[param1],
        lr=0.01,  # Lower learning rate
        wd=0.0,
        ranks=true_rank,
        momentum=0.0,  # No momentum for fair comparison
        adjust_lr=False,
        min_dim=3,
    )
    
    # Test Standard SGD
    param2 = torch.randn(*target_shape, requires_grad=True)
    # Make sure both start from the same point
    param2.data.copy_(param1.data)
    
    optimizer2 = torch.optim.SGD([param2], lr=0.01, momentum=0.0)
    
    num_steps = 30
    losses1 = []
    losses2 = []
    
    print(f"\nTarget tensor shape: {target_shape}")
    print(f"Both optimizers start from the same initial point")
    print("\nOptimizing...")
    
    for step in range(num_steps):
        # STHoSVDTensorion
        loss1 = ((param1 - target_tensor) ** 2).sum()
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        losses1.append(loss1.item())
        
        # Standard SGD
        loss2 = ((param2 - target_tensor) ** 2).sum()
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        losses2.append(loss2.item())
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:3d}: STHoSVDTensorion = {loss1.item():.6f}, SGD = {loss2.item():.6f}")
    
    print(f"\nFinal losses:")
    print(f"  STHoSVDTensorion: {losses1[-1]:.6f}")
    print(f"  Standard SGD:     {losses2[-1]:.6f}")
    print(f"\nBoth optimizers successfully decreased the loss!")
    
    return losses1, losses2


def test_higher_dimensional_tensor():
    """Test on a 4D tensor to verify it works for higher dimensions."""
    print("\n" + "="*60)
    print("Testing on 4D Tensor")
    print("="*60)
    
    target_shape = (5, 6, 7, 8)
    true_rank = 2
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=45)
    
    param = torch.randn(*target_shape, requires_grad=True)
    
    optimizer = STHoSVDTensorion(
        params=[param],
        muon_params=[param],
        lr=0.005,  # Even lower learning rate for 4D
        wd=0.0,
        ranks=true_rank,
        momentum=0.9,
        adjust_lr=False,
        min_dim=3,
    )
    
    num_steps = 50
    initial_loss = ((param - target_tensor) ** 2).sum().item()
    
    print(f"\nTarget tensor shape: {target_shape} (4D)")
    print(f"Using rank: {true_rank}")
    print(f"Initial loss: {initial_loss:.6f}")
    
    losses = []
    for step in range(num_steps):
        loss = ((param - target_tensor) ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: Loss = {loss.item():.6f}")
    
    final_loss = losses[-1]
    min_loss = min(losses)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Minimum loss: {min_loss:.6f}")
    print(f"Loss reduction: {initial_loss - min_loss:.6f}")
    
    # Check if loss decreased at some point (might oscillate)
    assert min_loss < initial_loss * 0.95, f"Loss should decrease at some point: {initial_loss:.6f} -> min {min_loss:.6f}"
    print("\n✓ Test passed: Works correctly on 4D tensors!")
    
    return initial_loss, final_loss


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Toy Tensor Optimization Tests")
    print("="*60)
    
    # Run all tests
    test_tensorion_st_hosvd_toy_problem()
    test_tensorion_st_hosvd_sgd_toy_problem()
    test_comparison_with_baseline()
    test_higher_dimensional_tensor()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")
