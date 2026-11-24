# Tensorion

**Muon for tensor layers.** Tensorion is a drop-in replacement for the [Muon](https://kellerjordan.github.io/posts/muon/) optimizer that also works on convolution kernels and other weights with more than two dimensions. You choose how each tensor is unfolded into a matrix before it is orthogonalized.

```python
from tensorion import Tensorion

optimizer = Tensorion(model.parameters(), lr=1e-3)
```

That is the whole setup. Matrices and tensors get the orthogonalized (Muon-style) update. Biases and norm parameters get AdamW. No parameter splitting or second optimizer is needed.

## Installation

```bash
pip install git+https://github.com/MTML-LAB/Tensorion.git
```

The only dependencies are PyTorch (≥ 2.1) and NumPy. Python ≥ 3.9.

## Switching from Muon

| You have | Replace it with |
|---|---|
| `torch.optim.Muon(matrix_params, ...)` + `torch.optim.AdamW(other_params, ...)` | `Tensorion(model.parameters(), ...)` |
| `MuonWithAuxAdam(param_groups)` with `use_muon` groups | `Tensorion(param_groups)`. The same groups work, including `betas` and `eps`. |

For a 2-D weight the update is exactly Muon's. A tensor of shape `(n_0, ..., n_{d-1})` is first unfolded into a matrix. It is orthogonalized like a Muon matrix and then folded back.

The defaults are the settings used in the paper. To reproduce `torch.optim.Muon` on matrices, use:

```python
Tensorion(params, nesterov=True, orthogonalizer="newton_schulz", adjust_lr_fn="original")
```

## Choosing the unfolding

The `unfolding` argument decides which axes of a tensor become rows. The remaining axes become columns.

| `unfolding=` | Rows of the matrix | Example for a conv kernel `(64, 32, 3, 3)` |
|---|---|---|
| `"min"` (default) | the smallest axis | `3 × 6144` |
| `"max"` | the largest axis | `64 × 288` |
| `"maxrank"` | the split of axes that allows the largest rank | `192 × 96` (axes 0 and 2) |
| `"dual"` | the split with the largest nuclear norm of the current update, chosen every step | depends on the update |
| `k` | axis `k` (negative indices work) | `unfolding=0` gives `64 × 288` |
| `(i, j, ...)` | the given axes | `unfolding=(0, 1)` gives `2048 × 9` |
| a function `f(update) -> axis or axes` | whatever you return | |

`"dual"` costs one SVD per candidate split at every step, so it is the slowest option.

Different layers can use different unfoldings through parameter groups:

```python
optimizer = Tensorion(
    [
        {"params": conv_params, "unfolding": "maxrank"},
        {"params": linear_params},  # uses the default below
    ],
    unfolding="min",
)
```

## Choosing the orthogonalizer

The `orthogonalizer` argument computes the polar factor `U Vᵀ` of the unfolded matrix `U S Vᵀ`.

| `orthogonalizer=` | Method | Matmuls per step | Converged after |
|---|---|---|---|
| `"svd"` (default) | exact SVD | | exact |
| `"newton_schulz"` | quintic Newton–Schulz iteration from Muon | 3 | never: singular values stay in about [0.7, 1.2] by design |
| `"polar_express"` | [Polar Express](https://arxiv.org/abs/2505.16932) | 3 | 8 steps (24 matmuls) |
| `"cans"` | [CANS](https://github.com/GrishKate/accelerating_orthogonalization), Chebyshev-accelerated Newton–Schulz with degree-3 polynomials | 2 | 10 steps (20 matmuls) |
| a function `f(matrix, steps) -> matrix` | your own | | |

The iterative methods use only matrix products, which run well on GPUs. Their iteration count is set by `ns_steps` (default 5). Compare them by matmuls, not steps: for the same number of matmuls, CANS is the most accurate. The "converged" column is for matrices with condition number up to 10³ (see `tests/test_orthogonalize.py`). CANS assumes normalized singular values of at least `1e-3`. For better-conditioned gradients, `functools.partial(tensorion.orthogonalize.cans, lower=...)` with a larger `lower` converges in fewer steps.

## All arguments

| Argument | Default | Meaning |
|---|---|---|
| `lr` | `1e-3` | learning rate |
| `weight_decay` | `0.1` | decoupled weight decay (coupled for the SGD fallback) |
| `momentum` | `0.95` | momentum of the Tensorion update |
| `nesterov` | `False` | Nesterov momentum |
| `unfolding` | `"min"` | see [Choosing the unfolding](#choosing-the-unfolding) |
| `orthogonalizer` | `"svd"` | see [Choosing the orthogonalizer](#choosing-the-orthogonalizer) |
| `ns_steps` | `5` | iterations of iterative orthogonalizers |
| `adjust_lr_fn` | `"match_rms_adamw"` | learning rate scaling for an unfolded `rows × cols` update: `"match_rms_adamw"` multiplies by `0.2·√max(rows, cols)` so AdamW learning rates carry over, `"original"` by `√max(1, rows/cols)`, `None` applies no scaling |
| `nuclear` | `False` | multiply the update by the nuclear norm of the unfolded momentum |
| `min_ndim` | `2` | parameters with at least this many dimensions get the Tensorion update. Use `3` to keep linear layers on AdamW. |
| `fallback` | `"adamw"` | update for the other parameters: `"adamw"` or `"sgd"` |
| `adamw_betas`, `adamw_eps` | `(0.9, 0.95)`, `1e-8` | AdamW fallback hyperparameters |
| `sgd_momentum` | `0.95` | SGD fallback momentum |

Every argument can be overridden per parameter group. A group can also set `use_muon=True/False` to force the update rule regardless of `min_ndim`.

## Example

```bash
python examples/quickstart.py
```

trains a small CNN on random data on the CPU in a few seconds.

## Reproducing the paper

The training pipeline (Hydra configs, ResNet/ViT models, CIFAR, Food-101 and ImageNet loaders) lives in [`experiments/`](experiments/). The exact code state used for the NeurIPS 2026 paper is tagged `neurips-2026`.

## Development

```bash
pip install -e ".[test]"
pytest
```
