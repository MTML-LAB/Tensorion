"""
Comprehensive comparison of all Tensorion optimizers:
- Tensorion (default, uses spectral sphere projection)
- TensorionSGD (uses spectral sphere projection with SGD backup)
- STHoSVDTensorion (uses ST-HOSVD truncation with Tucker decomposition)
- STHoSVDTensorionSGD (uses ST-HOSVD truncation with SGD backup)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorion.optimizers import (
    Tensorion, 
    TensorionSGD, 
    STHoSVDTensorion, 
    STHoSVDTensorionSGD
)
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


def optimize_and_track(target_tensor, optimizer_class, optimizer_kwargs, num_steps=100, seed=None):
    """Run optimization and track loss history."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
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
    
    final_param = param.data.clone()
    return losses, final_param


def run_comparison():
    """Run comprehensive comparison of all optimizers."""
    print("="*80)
    print("Tensorion Optimizers Comparison")
    print("="*80)
    
    # Create target tensor with known low rank
    target_shape = (8, 10, 12)
    true_rank = 3
    target_tensor, _, _ = create_low_rank_tensor(target_shape, true_rank, seed=42)
    
    print(f"\nTarget tensor shape: {target_shape}")
    print(f"True Tucker rank: {true_rank}")
    print(f"Target tensor norm: {torch.norm(target_tensor).item():.4f}")
    
    # Common hyperparameters
    base_lr = 0.01
    num_steps = 100
    
    # Define all optimizers with similar hyperparameters
    optimizers_config = {
        'Tensorion': {
            'class': Tensorion,
            'kwargs': {
                'lr': base_lr,
                'wd': 0.0,
                'momentum': 0.9,
                'nesterov': False,
                'nuclear': False,
                'adjust_lr': False,
                'min_dim': 3,
            }
        },
        'TensorionSGD': {
            'class': TensorionSGD,
            'kwargs': {
                'lr': base_lr,
                'wd': 0.0,
                'momentum': 0.9,
                'nesterov': False,
                'nuclear': False,
                'adjust_lr': False,
                'min_dim': 3,
            }
        },
        'STHoSVDTensorion': {
            'class': STHoSVDTensorion,
            'kwargs': {
                'lr': base_lr,
                'wd': 0.0,
                'ranks': true_rank,
                'momentum': 0.9,
                'nesterov': False,
                'nuclear': False,
                'adjust_lr': False,
                'min_dim': 3,
            }
        },
        'STHoSVDTensorionSGD': {
            'class': STHoSVDTensorionSGD,
            'kwargs': {
                'lr': base_lr,
                'wd': 0.0,
                'ranks': true_rank,
                'momentum': 0.9,
                'nesterov': False,
                'nuclear': False,
                'adjust_lr': False,
                'min_dim': 3,
            }
        },
    }
    
    print("\n" + "-"*80)
    print("Running optimizations...")
    print("-"*80)
    
    results = {}
    seed_base = 100
    
    for name, config in optimizers_config.items():
        print(f"\nOptimizing with {name}...")
        opt_class = config['class']
        kwargs = config['kwargs']
        
        losses, final_param = optimize_and_track(
            target_tensor, 
            opt_class, 
            kwargs, 
            num_steps=num_steps,
            seed=seed_base
        )
        
        results[name] = {
            'losses': losses,
            'final_param': final_param,
        }
        
        final_loss = losses[-1]
        min_loss = min(losses)
        relative_error = torch.norm(final_param - target_tensor).item() / torch.norm(target_tensor).item()
        reduction = (losses[0] - min_loss) / losses[0] * 100
        
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Minimum loss: {min_loss:.4f}")
        print(f"  Loss reduction: {reduction:.2f}%")
        print(f"  Relative error: {relative_error:.4f}")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Loss curves
        ax1 = axes[0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (name, data) in enumerate(results.items()):
            ax1.plot(data['losses'], label=name, alpha=0.8, linewidth=2, color=colors[i])
        
        ax1.set_xlabel('Optimization Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Loss reduction percentage
        ax2 = axes[1]
        names = list(results.keys())
        reductions = [(results[n]['losses'][0] - min(results[n]['losses'])) / results[n]['losses'][0] * 100 
                     for n in names]
        bars = ax2.bar(names, reductions, color=colors, alpha=0.7)
        ax2.set_ylabel('Loss Reduction (%)', fontsize=12)
        ax2.set_title('Final Loss Reduction', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 105])
        
        # Add value labels on bars
        for bar, val in zip(bars, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = '/home/sakudryashov/tensorion/Tensorion/tests/unit/tensorion_optimizers_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"\nCould not create plot (matplotlib may not be available): {e}")
    
    # Print detailed comparison table
    print("\n" + "="*80)
    print("Detailed Comparison Table")
    print("="*80)
    print(f"{'Optimizer':<25} {'Initial':>12} {'Final':>12} {'Min':>12} {'Reduction':>12} {'Rel Error':>12}")
    print("-"*80)
    
    for name, data in results.items():
        losses = data['losses']
        initial = losses[0]
        final = losses[-1]
        min_loss = min(losses)
        reduction = (initial - min_loss) / initial * 100
        rel_error = torch.norm(data['final_param'] - target_tensor).item() / torch.norm(target_tensor).item()
        
        print(f"{name:<25} {initial:>12.2f} {final:>12.2f} {min_loss:>12.2f} {reduction:>11.2f}% {rel_error:>11.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    # Find best optimizer by minimum loss
    best_by_min = min(results.items(), key=lambda x: min(x[1]['losses']))
    print(f"Best minimum loss: {best_by_min[0]} ({min(best_by_min[1]['losses']):.4f})")
    
    # Find best optimizer by final loss
    best_by_final = min(results.items(), key=lambda x: x[1]['losses'][-1])
    print(f"Best final loss: {best_by_final[0]} ({best_by_final[1]['losses'][-1]:.4f})")
    
    # Find best optimizer by relative error
    best_by_error = min(results.items(), 
                       key=lambda x: torch.norm(x[1]['final_param'] - target_tensor).item() / 
                                     torch.norm(target_tensor).item())
    best_error = torch.norm(best_by_error[1]['final_param'] - target_tensor).item() / \
                 torch.norm(target_tensor).item()
    print(f"Best relative error: {best_by_error[0]} ({best_error:.4f})")
    
    # Convergence speed (steps to reach 50% of initial loss)
    print("\nConvergence speed (steps to 50% reduction):")
    for name, data in results.items():
        initial = data['losses'][0]
        target = initial * 0.5
        steps_to_50 = next((i for i, loss in enumerate(data['losses']) if loss < target), num_steps)
        print(f"  {name:<25}: {steps_to_50:>3d} steps")
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80)
    
    return results


def compare_on_different_shapes():
    """Compare optimizers on different tensor shapes."""
    print("\n\n" + "="*80)
    print("Comparison on Different Tensor Shapes")
    print("="*80)
    
    shapes = [
        ((6, 8, 10), 2, "Small 3D"),
        ((8, 10, 12), 3, "Medium 3D"),
        ((5, 6, 7, 8), 2, "4D Tensor"),
    ]
    
    base_lr = 0.01
    num_steps = 50
    
    for shape, rank, desc in shapes:
        print(f"\n{desc}: shape={shape}, rank={rank}")
        print("-"*80)
        
        target_tensor, _, _ = create_low_rank_tensor(shape, rank, seed=42)
        
        optimizers = {
            'Tensorion': (Tensorion, {'lr': base_lr, 'wd': 0.0, 'momentum': 0.9, 'adjust_lr': False, 'min_dim': 3}),
            'TensorionSGD': (TensorionSGD, {'lr': base_lr, 'wd': 0.0, 'momentum': 0.9, 'adjust_lr': False, 'min_dim': 3}),
            'STHoSVDTensorion': (STHoSVDTensorion, {'lr': base_lr, 'wd': 0.0, 'ranks': rank, 'momentum': 0.9, 'adjust_lr': False, 'min_dim': 3}),
            'STHoSVDTensorionSGD': (STHoSVDTensorionSGD, {'lr': base_lr, 'wd': 0.0, 'ranks': rank, 'momentum': 0.9, 'adjust_lr': False, 'min_dim': 3}),
        }
        
        for name, (opt_class, kwargs) in optimizers.items():
            losses, _ = optimize_and_track(target_tensor, opt_class, kwargs, num_steps=num_steps, seed=100)
            reduction = (losses[0] - min(losses)) / losses[0] * 100
            print(f"  {name:<25}: {losses[0]:>8.2f} → {min(losses):>8.2f} ({reduction:>5.1f}% reduction)")


if __name__ == "__main__":
    # Main comparison
    results = run_comparison()
    
    # Additional comparison on different shapes
    compare_on_different_shapes()
    
    print("\n" + "="*80)
    print("All comparisons completed!")
    print("="*80 + "\n")
