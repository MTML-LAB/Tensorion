# Tensorion

**Muon for tensor layers.** Tensorion is a drop-in replacement for the [Muon](https://kellerjordan.github.io/posts/muon/) optimizer that also works on convolution kernels and other weights with more than two dimensions. You choose how each tensor is unfolded into a matrix before it is orthogonalized.

Paper: [Tensorion: A Tensor-Aware Generalization of the Muon Optimizer](https://arxiv.org/abs/2606.25975), NeurIPS 2026.

```python
from tensorion import Tensorion

optimizer = Tensorion(model.parameters(), lr=1e-3)
```

That is the whole setup. Matrices and tensors get the orthogonalized (Muon-style) update. Biases and norm parameters get AdamW. No parameter splitting or second optimizer is needed.

## Installation

```bash
pip install git+https://github.com/MTML-LAB/Tensorion.git
```

Requires Python ≥ 3.12. The only dependencies are PyTorch (≥ 2.15) and NumPy (≥ 2.5). There are no upper version bounds, so installing Tensorion never downgrades anything.

## Switching from Muon

| You have | Replace it with |
|---|---|
| `torch.optim.Muon(matrix_params, ...)` + `torch.optim.AdamW(other_params, ...)` | `Tensorion(model.parameters(), ...)` |
| `MuonWithAuxAdam(param_groups)` with `use_muon` groups | `Tensorion(param_groups)`. The same groups work, including `betas` and `eps`. |

For a 2-D weight the update is exactly Muon's. A tensor of shape `(n_0, ..., n_{d-1})` is first unfolded into a matrix. It is orthogonalized like a Muon matrix and then folded back.

The defaults follow Algorithm 1 of the paper: the offline unfolding, Newton–Schulz orthogonalization, and the AdamW-matched learning rate scaling. To reproduce `torch.optim.Muon` on matrices, use:

```python
Tensorion(params, nesterov=True, adjust_lr_fn="original")
```

## Choosing the unfolding

The `unfolding` argument decides which axes of a tensor become rows. The remaining axes become columns. The paper's two strategies work for any shape:

| `unfolding=` | Rows of the matrix | Example for a conv kernel `(64, 32, 3, 3)` |
|---|---|---|
| `"offline"` (default) | the split of axes that allows the largest rank, fixed by the shape | `192 × 96` (axes 0 and 2) |
| `"online"` | the split with the largest nuclear norm of the current update, chosen again during training | depends on the update |
| `"online_exact"` | the same choice, with nuclear norms from exact SVDs | depends on the update |

`"offline"` is the paper's main strategy. `"online"` picks the unfolding that is optimal for the paper's tensor norm.

### How the online choice stays cheap

Taking an SVD of every unfolding at every step is expensive. `"online"` avoids all of them:

- **Most splits are never formed.** All unfoldings of a tensor have the same Frobenius norm, and the nuclear norm of a matrix is at most `√rank` times it. Splits are visited from the largest rank down, and the search stops once this bound falls below the best split so far. On a CIFAR ResNet-18, 2 of the 7 splits of a 3×3 kernel are looked at in most layers and 4 at most. A 1×1 convolution or a matrix has a single unfolding and costs nothing.
- **Nuclear norms are estimated with matrix products only.** The estimate is `tr((MMᵀ)^½)` by stochastic Lanczos quadrature. It averages the Gauss and Gauss–Radau rules, which over- and under-estimate, and it uses [Hutch++](https://arxiv.org/abs/2010.09649) deflation to cut the probe noise for low-rank gradients. See [`tensorion/nuclear.py`](src/tensorion/nuclear.py).
- **The choice is reused.** The split is selected every `online_every` steps (10 by default) and kept in between. The momentum changes slowly, so this costs little. `unfolding="online_exact", online_every=1` is the paper's online strategy: exact, at every step.

Measured on momentum tensors of a CIFAR ResNet-18 and on synthetic tensors (CPU, `python benchmarks/online_selection.py` in `experiments/`):

| | `"online_exact"`, every step | `"online"`, every step | `"online"`, every 10 steps (default) |
|---|---|---|---|
| Time of `optimizer.step()` over `"offline"` | about +350% | about +30% | within noise (±4%) |
| Nuclear norm lost against the best split | 0 | 0.04% on average, 0.5% at most | 0.04% on average, 0.7% at most |

The estimate of a single nuclear norm is off by 1% on average and 5% at most. When `"online"` picks a different split than the exact per-step choice, the two splits are near-ties, so the loss stays small. Selecting every 20 steps loses the same amount. The probes are fixed, so the choice is deterministic and does not touch the global random state.

### Per-layer unfoldings

A fixed unfolding only makes sense for a particular shape, so it is set per layer. Pass `model.named_parameters()` and map parameter names, or [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) patterns, to a strategy, an axis, or a tuple of axes:

```python
optimizer = Tensorion(
    model.named_parameters(),
    unfolding={
        "conv1.weight": (0, 1),           # rows: axes 0 and 1  ->  (64·3) × (7·7)
        "layer*.0.conv1.weight": 0,       # rows: axis 0
        "layer4.*.weight": "online",
    },
)
```

An exact name wins over patterns, and patterns are tried in order. Layers that match nothing use `"offline"`. Every layer's unfolding is checked against its shape when the optimizer is built, so a wrong axis fails right away with the layer's name. A key that matches no layer is reported too.

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
| `unfolding` | `"offline"` | `"offline"`, `"online"`, `"online_exact"`, or a per-layer mapping, see [Choosing the unfolding](#choosing-the-unfolding) |
| `online_every` | `10` | with an online unfolding, choose the split every this many steps and reuse it in between. `1` selects at every step. |
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

## JAX (experimental)

The same optimizer is available for JAX as [optax](https://github.com/google-deepmind/optax) gradient transformations:

```bash
pip install "tensorion[jax] @ git+https://github.com/MTML-LAB/Tensorion.git"
```

```python
import optax, tensorion_jax

optimizer = tensorion_jax.tensorion(learning_rate=1e-3)
state = optimizer.init(params)
updates, state = optimizer.update(grads, state, params)
params = optax.apply_updates(params, updates)
```

`tensorion_jax.tensorion` takes the same arguments with the same defaults as the PyTorch `Tensorion`, and `learning_rate` can be an optax schedule. The differences:

- Per-layer unfoldings are keyed by the path of a parameter in the pytree, joined by `/`: `unfolding={"conv1/kernel": (0, 1)}`.
- The AdamW betas are `adamw_b1` and `adamw_b2`, and `param_labels` replaces `use_muon` groups.
- `tensorion_jax.scale_by_tensorion` is the bare transformation, to combine with other optax pieces.

Everything works under `jax.jit`, including the online unfolding. The tests check that it produces the same weights as the PyTorch optimizer, step for step. See [`examples/quickstart_jax.py`](examples/quickstart_jax.py).

The stock `optax.contrib.muon` can already use the offline unfolding, without the rest of this package:

```python
optax.contrib.muon(1e-3, muon_weight_dimension_numbers=tensorion_jax.offline_dimension_numbers)
```

## Reproducing the paper

The training pipeline (Hydra configs, ResNet/ViT models, CIFAR, Food-101 and ImageNet loaders) lives in [`experiments/`](experiments/). The release that accompanies the NeurIPS 2026 paper is tagged `neurips-2026`.

## Citation

```bibtex
@inproceedings{bogachev2026tensorion,
  title         = {Tensorion: A Tensor-Aware Generalization of the Muon Optimizer},
  author        = {Bogachev, Vladimir and Aletov, Vladimir and Molozhavenko, Alexander and Kudriashov, Sergei and Rakhuba, Maxim},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2606.25975},
  archivePrefix = {arXiv}
}
```

## License

[MIT](LICENSE)

## Development

```bash
pip install -e ".[test]"
pytest
```

The JAX tests are skipped unless `jax` and `optax` are installed (`pip install -e ".[test,jax]"`).
