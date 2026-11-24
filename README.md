# Tensorion

**Muon for tensor layers.** Tensorion is a drop-in replacement for the [Muon](https://kellerjordan.github.io/posts/muon/) optimizer that also works on convolution kernels and other weights with more than two dimensions. You choose how each tensor is unfolded into a matrix before it is orthogonalized.

Paper: [Tensorion: A Tensor-Aware Generalization of the Muon Optimizer](https://arxiv.org/abs/2606.25975).

```python
from tensorion import Tensorion

optimizer = Tensorion(model.parameters(), lr=1e-3)
```

That is the whole setup. Matrices and tensors get the orthogonalized (Muon-style) update. Biases and norm parameters get AdamW. No parameter splitting or second optimizer is needed.

## Installation

```bash
pip install git+https://github.com/MTML-LAB/Tensorion.git
```

The only dependencies are PyTorch (≥ 2.1) and NumPy, with no upper version bounds, so installing Tensorion never downgrades them. Python ≥ 3.9.

## Switching from Muon

| You have | Replace it with |
|---|---|
| `torch.optim.Muon(matrix_params, ...)` + `torch.optim.AdamW(other_params, ...)` | `Tensorion(model.parameters(), ...)` |
| `MuonWithAuxAdam(param_groups)` with `use_muon` groups | `Tensorion(param_groups)`. The same groups work, including `betas` and `eps`. |

For a 2-D weight the update is exactly Muon's. A tensor of shape `(n_0, ..., n_{d-1})` is first unfolded into a matrix. It is orthogonalized like a Muon matrix and then folded back.

The defaults follow Algorithm 1 of the paper: the max-rank ("offline") unfolding, Newton–Schulz orthogonalization, and the AdamW-matched learning rate scaling. To reproduce `torch.optim.Muon` on matrices, use:

```python
Tensorion(params, nesterov=True, adjust_lr_fn="original")
```

## Choosing the unfolding

The `unfolding` argument decides which axes of a tensor become rows. The remaining axes become columns.

| `unfolding=` | Rows of the matrix | Example for a conv kernel `(64, 32, 3, 3)` |
|---|---|---|
| `"maxrank"` or `"offline"` (default) | the split of axes that allows the largest rank, fixed by the shape | `192 × 96` (axes 0 and 2) |
| `"dual"` or `"online"` | the split with the largest nuclear norm of the current update, chosen every step | depends on the update |
| `"min"` | the smallest axis | `3 × 6144` |
| `"max"` | the largest axis | `64 × 288` |
| `k` | axis `k` (negative indices work) | `unfolding=0` gives `64 × 288` |
| `(i, j, ...)` | the given axes | `unfolding=(0, 1)` gives `2048 × 9` |
| a function `f(update) -> axis or axes` | whatever you return | |

`"maxrank"` is the paper's main strategy. `"dual"` picks the unfolding that is optimal for the paper's tensor norm, but it costs one SVD per candidate split at every step, so it is the slowest option.

Different layers can use different unfoldings through parameter groups:

```python
optimizer = Tensorion(
    [
        {"params": conv_params, "unfolding": "dual"},
        {"params": linear_params},  # uses the default below
    ],
    unfolding="maxrank",
)
```

## Choosing the orthogonalizer

The `orthogonalizer` argument computes the polar factor `U Vᵀ` of the unfolded matrix `U S Vᵀ`.

| `orthogonalizer=` | Method | Matmuls per step | Default steps | Converged after |
|---|---|---|---|---|
| `"newton_schulz"` (default) | quintic Newton–Schulz iteration from Muon | 3 | 5 | never: singular values stay in about [0.7, 1.2] by design |
| `"polar_express"` | Polar Express ([paper](https://arxiv.org/abs/2505.16932), [code](https://github.com/NoahAmsel/PolarExpress)) | 3 | 5 | 8 steps (24 matmuls) |
| `"cans"` | CANS, Chebyshev-accelerated Newton–Schulz with degree-3 polynomials ([paper](https://arxiv.org/abs/2506.10935), [code](https://github.com/GrishKate/accelerating_orthogonalization)) | 2 | 7 | 10 steps (20 matmuls) |
| `"svd"` | exact SVD | | | exact |
| a function `f(matrix, steps) -> matrix` | your own | | | |

The iterative methods use only matrix products, which run well on GPUs. The default step counts cost about 15 matmuls for each method. `ns_steps` overrides them. Compare the methods by matmuls, not steps: for the same number of matmuls, CANS is the most accurate. The "converged" column is for matrices with condition number up to 10³ (see `tests/test_orthogonalize.py`).

The CANS defaults are the Muon setting of the CANS paper (7 steps, lower singular value bound `9e-4`), which map singular values into about [0.7, 1.3] like Newton–Schulz. For better-conditioned gradients, `functools.partial(tensorion.orthogonalize.cans, lower=...)` with a larger `lower` converges in fewer steps.

## All arguments

| Argument | Default | Meaning |
|---|---|---|
| `lr` | `1e-3` | learning rate |
| `weight_decay` | `0.1` | decoupled weight decay (coupled for the SGD fallback) |
| `momentum` | `0.95` | momentum of the Tensorion update |
| `nesterov` | `False` | Nesterov momentum |
| `unfolding` | `"maxrank"` | see [Choosing the unfolding](#choosing-the-unfolding) |
| `orthogonalizer` | `"newton_schulz"` | see [Choosing the orthogonalizer](#choosing-the-orthogonalizer) |
| `ns_steps` | `None` | iterations of iterative orthogonalizers. `None` uses each method's default. |
| `adjust_lr_fn` | `"match_rms_adamw"` | learning rate scaling for an unfolded `rows × cols` update: `"match_rms_adamw"` multiplies by `0.2·√max(rows, cols)` so AdamW learning rates carry over, `"original"` by `√max(1, rows/cols)`, `None` applies no scaling |
| `nuclear` | `False` | multiply the update by the nuclear norm of the unfolded momentum |
| `min_ndim` | `2` | parameters with at least this many dimensions get the Tensorion update. Use `3` to keep linear layers on AdamW, as in the paper's experiments. |
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

## Citation

```bibtex
@misc{bogachev2026tensorion,
  title         = {Tensorion: A Tensor-Aware Generalization of the Muon Optimizer},
  author        = {Bogachev, Vladimir and Aletov, Vladimir and Molozhavenko, Alexander and Kudriashov, Sergei and Rakhuba, Maxim},
  year          = {2026},
  eprint        = {2606.25975},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Development

```bash
pip install -e ".[test]"
pytest
```
