"""
Visualize tensor optimization progress with TensorionSTHoSVD.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from tensorion.optimizers import TensorionSTHoSVD, TensorionSTHoSVDSGD
from tensorion.tensor_utils import tucker_reconstruct


def create_low_rank_tensor(shape, rank, seed=42):
    """Create a tensor with low Tucker rank for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    factors = []
    core_shape = [rank] * len(shape)
    core = torch.randn(*core_shape)
    
    for i, dim in enumerate(shape):
        factor = torch.randn(dim, rank)
        factors.append(factor)
    
    tensor = tucker_reconstruct(core, factors)
    return tensor, core, factors


def optimize_and_track(target_tensor, optimizer_class, optimizer_kwargs, num_steps=100):
    """Run optimization and track loss history."""
    param = torch.randn_like(target_tensor, requires_grad=True)
    
    optimizer = optimizer_class(
        params=[param],
        muon_params=[param],
        **optimizer_kwargs
    )
    
    losses = []
    for step in range(num_steps):
        loss = ((param - target_tensor) ** 2).sum()
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return losses, param


def main():
    """Run optimization and visualize results."""
    print("="*70)
    print("Tensor Optimization Visualization")
    print("="*70)
    
    # Create target tensor
    target_shape = (8, 10, 12)
    true_rank = 3
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=42)
    
    print(f"\nTarget tensor shape: {target_shape}")
    print(f"True Tucker rank: {true_rank}")
    print(f"Target tensor norm: {torch.norm(target_tensor).item():.4f}")
    
    # Test different optimizers
    optimizers = {
        'TensorionSTHoSVD': (TensorionSTHoSVD, {
            'lr': 0.01,
            'wd': 0.0,
            'ranks': true_rank,
            'momentum': 0.9,
            'adjust_lr': False,
            'min_dim': 3,
        }),
        'TensorionSTHoSVDSGD': (TensorionSTHoSVDSGD, {
            'lr': 0.01,
            'wd': 0.0,
            'ranks': true_rank,
            'momentum': 0.9,
            'adjust_lr': False,
            'min_dim': 3,
        }),
    }
    
    # Also compare with standard SGD
    print("\n" + "-"*70)
    print("Running optimizations...")
    print("-"*70)
    
    results = {}
    num_steps = 100
    
    for name, (opt_class, kwargs) in optimizers.items():
        print(f"\nOptimizing with {name}...")
        losses, final_param = optimize_and_track(target_tensor, opt_class, kwargs, num_steps)
        results[name] = losses
        
        final_loss = losses[-1]
        min_loss = min(losses)
        relative_error = torch.norm(final_param - target_tensor).item() / torch.norm(target_tensor).item()
        
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Minimum loss: {min_loss:.4f}")
        print(f"  Relative error: {relative_error:.4f}")
    
    # Standard SGD for comparison
    print(f"\nOptimizing with Standard SGD...")
    param_sgd = torch.randn_like(target_tensor, requires_grad=True)
    optimizer_sgd = torch.optim.SGD([param_sgd], lr=0.01, momentum=0.9)
    losses_sgd = []
    
    for step in range(num_steps):
        loss = ((param_sgd - target_tensor) ** 2).sum()
        losses_sgd.append(loss.item())
        loss.backward()
        optimizer_sgd.step()
        optimizer_sgd.zero_grad()
    
    results['Standard SGD'] = losses_sgd
    print(f"  Initial loss: {losses_sgd[0]:.4f}")
    print(f"  Final loss: {losses_sgd[-1]:.4f}")
    print(f"  Minimum loss: {min(losses_sgd):.4f}")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 6))
        
        for name, losses in results.items():
            plt.plot(losses, label=name, alpha=0.8, linewidth=2)
        
        plt.xlabel('Optimization Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Tensor Optimization: Low-Rank Approximation', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        # Save plot
        plot_path = '/home/sakudryashov/tensorion/Tensorion/tests/unit/tensor_optimization_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"\nCould not create plot (matplotlib may not be available): {e}")
        print("But optimization completed successfully!")
    
    print("\n" + "="*70)
    print("Optimization completed!")
    print("="*70)
    
    # Summary statistics
    print("\nSummary:")
    print("-"*70)
    for name, losses in results.items():
        reduction = (losses[0] - min(losses)) / losses[0] * 100
        print(f"{name:25s}: {losses[0]:8.2f} → {min(losses):8.2f} ({reduction:5.1f}% reduction)")


if __name__ == "__main__":
    main()
