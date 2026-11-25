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
| `torch.optim.Muon(matrix_params, ...)` + `torch.optim.AdamW(other_params, ...)` | `Tensorion(model.parameters(), ...)`, or `Tensorion.from_model(model, ...)` to keep the head and embeddings on AdamW |
| `MuonWithAuxAdam(param_groups)` with `use_muon` groups | `Tensorion(param_groups)`. The same groups work, including `betas` and `eps`. |

For a 2-D weight the update is exactly Muon's. A tensor of shape `(n_0, ..., n_{d-1})` is first unfolded into a matrix. It is orthogonalized like a Muon matrix and then folded back.

The defaults follow Algorithm 1 of the paper: the offline unfolding, Newton–Schulz orthogonalization, and the AdamW-matched learning rate scaling. To reproduce `torch.optim.Muon` on matrices, use:

```python
Tensorion(params, nesterov=True, adjust_lr_fn="original")
```

## Heads, embeddings and other special layers

`Tensorion(model.parameters())` routes by shape alone: every parameter with at least two axes larger than 1 gets the Tensorion update. The optimizer cannot tell an output head, an embedding table or a positional embedding from any other matrix, and such layers are usually trained with AdamW in Muon setups. There are two ways to keep them there.

`Tensorion.from_model` routes by module type instead of by shape:

```python
optimizer = Tensorion.from_model(model, lr=1e-3)
```

The weights of linear layers, convolutions and attention projections get the Tensorion update. Embedding tables, bare parameters such as positional embeddings and class tokens, norms and biases use the fallback, and so does the output head. The head is taken to be the last linear layer of the model. Name it with `head="lm_head"` if this guess is wrong, or pass `head=None` to train it with Tensorion as well. The `modules` argument changes the module classes, e.g. `modules=(torch.nn.Conv2d,)` for the routing of the paper.

With `exclude`, the routing stays by shape and the listed names or `fnmatch` patterns go to the fallback:

```python
optimizer = Tensorion(model.named_parameters(), exclude=["fc.*", "*embed*"])
```

In both cases `optimizer.routing()` returns the update rule of every parameter, so the result can be checked before training:

```python
>>> optimizer.routing()
{'conv1.weight': 'tensorion', 'bn1.weight': 'adamw', ..., 'fc.weight': 'adamw', 'fc.bias': 'adamw'}
```

Parameter groups with `use_muon=True/False` remain the fully manual option.

## Choosing the unfolding

The `unfolding` argument decides which axes of a tensor become rows. The remaining axes become columns. The paper's two strategies work for any shape:

| `unfolding=` | Rows of the matrix | Example for a conv kernel `(64, 32, 3, 3)` |
|---|---|---|
| `"offline"` (default) | the split of axes that allows the largest rank, fixed by the shape | `192 × 96` (axes 0 and 2) |
| `"online"` | the split with the largest nuclear norm of the current update, chosen again during training | depends on the update |
| `"online_polar"` | the same choice, with nuclear norms from the polar factors, and the winner's factor reused as the update | depends on the update |
| `"online_exact"` | the same choice, with nuclear norms from exact SVDs | depends on the update |

`"offline"` is the paper's main strategy. `"online"` picks the unfolding that is optimal for the paper's tensor norm.

Axes of size 1 are ignored everywhere, so the optimizer does not depend on how a weight is stored. A depthwise kernel `(64, 1, 3, 3)` has the unfoldings of a `(64, 3, 3)` tensor. A 1×1 convolution `(128, 64, 1, 1)` is a matrix and gets the same update as a linear layer with the same weights. A parameter with fewer than two axes larger than 1, such as a class token `(1, 1, 768)`, is a vector and goes to the fallback optimizer.

### How the online choice stays cheap

Taking an SVD of every unfolding at every step is expensive. Three things avoid it:

- **Most splits are never formed.** All unfoldings of a tensor have the same Frobenius norm, and the nuclear norm of a matrix is at most `√rank` times it. Splits are visited from the largest rank down, and the search stops once this bound falls below the best split so far. On a CIFAR ResNet, 2 of the 7 splits of a 3×3 kernel are looked at in most layers and 4 at most. A 1×1 convolution or a matrix has a single unfolding and costs nothing.
- **Nuclear norms come from matrix products.** There are two ways, both without an SVD:
  - `"online_polar"` uses `‖M‖_* = ⟨M, P⟩` for the polar factor `P` of `M`, computed by the optimizer's own orthogonalizer. The factor of the winning split is reused as the update, so only the losing candidates cost extra. It needs matrix products and nothing else.
  - `"online"` estimates `tr((MMᵀ)^½)` by stochastic Lanczos quadrature. It averages the Gauss and Gauss–Radau rules, which over- and under-estimate, and uses [Hutch++](https://arxiv.org/abs/2010.09649) deflation against probe noise. It needs far fewer floating point operations, but many small ones, plus `cholesky` and `eigvalsh` of tiny matrices. See [`tensorion/nuclear.py`](src/tensorion/nuclear.py).
- **The choice is reused.** The split is selected every `online_every` steps (10 by default) and kept in between. `unfolding="online_exact", online_every=1` is the paper's online strategy: exact, at every step.

Everything stays on the device of the parameters. One number per candidate is read back to decide which split wins. `matmul_dtype="bfloat16"` runs the large products of the orthogonalizer and of the selection in bfloat16, as Muon does.

Measured on the 36 conv kernels of a ResNet-34 on an Intel Arc A770 (`python benchmarks/online_variants.py` in `experiments/`):

| | `"offline"` | `"online_polar"` | `"online"` | `"online_exact"` |
|---|---|---|---|---|
| Nuclear norm lost against the best split, average / worst | 0.03% / 0.3% | 0.02% / 0.3% | 0.04% / 0.3% | 0 |
| One selection for all layers | none | 0.57 s | 1.45 s | 1.63 s |
| Time of `optimizer.step()` over `"offline"`, selecting every 10 steps | | +22% | +94% | +96% |
| Operations that PyTorch runs on the CPU on this backend | none | none | `cholesky`, `eigvalsh` | SVD |

Peak memory is the same for all of them. Two things stand out. The offline choice already loses almost nothing against the best split on this model, because the candidates that matter are near-ties. And which estimate is faster depends on the device: on this GPU, where every small operation has a fixed cost, `"online_polar"` is 2.5 times faster than the Lanczos estimate, while on a CPU the Lanczos estimate added about 30% to a step even when selecting at every step.

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
| `unfolding` | `"offline"` | `"offline"`, `"online"`, `"online_polar"`, `"online_exact"`, or a per-layer mapping, see [Choosing the unfolding](#choosing-the-unfolding) |
| `online_every` | `10` | with an online unfolding, choose the split every this many steps and reuse it in between. `1` selects at every step. |
| `orthogonalizer` | `"newton_schulz"` | see [Choosing the orthogonalizer](#choosing-the-orthogonalizer) |
| `ns_steps` | `None` | iterations of iterative orthogonalizers. `None` uses each method's default. |
| `matmul_dtype` | `None` | precision of the large matrix products of the orthogonalizer and the online selection, e.g. `"bfloat16"`. `None` uses the precision of the parameter. |
| `adjust_lr_fn` | `"match_rms_adamw"` | learning rate scaling for an unfolded `rows × cols` update: `"match_rms_adamw"` multiplies by `0.2·√max(rows, cols)` so AdamW learning rates carry over, `"original"` by `√max(1, rows/cols)`, `None` applies no scaling |
| `nuclear` | `False` | multiply the update by the nuclear norm of the unfolded momentum |
| `min_order` | `2` | parameters of at least this order get the Tensorion update. The order is the number of axes larger than 1. `3` keeps every matrix, including 1×1 convolutions, on the fallback. |
| `exclude` | `()` | parameter names or patterns that always use the fallback, see [Heads, embeddings and other special layers](#heads-embeddings-and-other-special-layers) |
| `fallback` | `"adamw"` | update for the other parameters: `"adamw"` or `"sgd"` |
| `adamw_betas`, `adamw_eps` | `(0.9, 0.95)`, `1e-8` | AdamW fallback hyperparameters |
| `sgd_momentum` | `0.95` | SGD fallback momentum |

Every argument can be overridden per parameter group. A group can also set `use_muon=True/False` to force the update rule regardless of `min_order`. `exclude` takes precedence over both.

## Example

```bash
python examples/quickstart.py
```

trains a small CNN on random data on the CPU in a few seconds.

## Reproducing the paper

The training pipeline (Hydra configs, ResNet/ViT models, CIFAR, Food-101 and ImageNet loaders) lives in [`experiments/`](experiments/). The experiments of the paper apply Tensorion to every convolution kernel and train all other parameters with the fallback. This includes 1×1 convolutions, which are matrices and therefore receive plain Muon updates. The experiment config states this routing explicitly, by module type. The release that accompanies the NeurIPS 2026 paper is tagged `neurips-2026`.

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
