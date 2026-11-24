# Experiments

The training pipeline used for the paper. It is not part of the `tensorion` package. It uses the optimizer like any other user would.

## Setup

From the repository root:

```bash
pip install -e .
pip install -r experiments/requirements.txt
cd experiments
```

## Running

```bash
python run.py                                          # ResNet-18 on CIFAR-10 with Tensorion
python run.py optimizer.unfolding=online               # another unfolding
python run.py optimizer.unfolding=online_exact optimizer.online_every=1  # the paper's online strategy
python run.py 'optimizer.unfolding={conv1.weight:[0,1]}'  # a fixed unfolding for one layer
python run.py optimizer.orthogonalizer=polar_express   # another orthogonalizer
python run.py optimizer=adamw dataset=cifar100         # a baseline
python run.py -m optimizer.lr=1e-4,3e-4,1e-3           # a sweep
```

Which layers get the Tensorion update is set in `configs/optimizer/tensorion.yaml` by `tensorion_modules`, a list of module classes. The default is every convolution kernel, including 1×1 convolutions, as in the paper. All other parameters use the fallback optimizer.

Datasets are downloaded to `./data` on first use. Results go to `./results/`.

Configs live in `configs/`:

| Group | Options |
|---|---|
| `optimizer` | `tensorion`, `adamw`, `sgd`, `shampoo` |
| `model` | `resnet18`, `vit` (needs `pip install fastervit`) |
| `dataset` | `cifar10`, `cifar100`, `food101`, `imagenet` |
| `scheduler` | `noscheduler`, `cosinelr`, `multistep`, `onecyclelr` |

To log to [Comet](https://www.comet.com/), `pip install comet_ml`, then pass `comet.enabled=true comet.api_key=... comet.workspace=...`.

## Tests

```bash
pytest -m "not integration"   # fast unit tests
TENSORION_TEST_DOWNLOAD=1 pytest -m integration   # one short CIFAR-10 epoch
```

`benchmarks/` holds three scripts: `online_selection.py` (the Lanczos estimate against exact selection, on CPU), `online_variants.py` (all online strategies on a GPU: selection quality, step time, memory) and `offline_vs_online.py` (a short training run per strategy, with time and peak memory per epoch).

`notebooks/` holds exploratory notebooks from development. They import modules by their old paths and are kept for reference only.
