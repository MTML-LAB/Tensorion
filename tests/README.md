# Tests

## Run

Unit + integration (default):
```bash
pytest -q
```

Unit only:
```bash
pytest -q tests/unit
```

Integration only:
```bash
pytest -q tests/integration/test_resnet18_cifar10_training.py
```

## Data for integration

The integration test runs a short ResNet18 + CIFAR10 training loop on CPU.

Environment variables:
- `TENSORION_DATA_DIR`: dataset root (defaults to `./data`)
- `TENSORION_TEST_DOWNLOAD=1`: allow CIFAR10 download if not present

The integration test uses small CIFAR10 subsets (64 train / 64 test) and
`num_epochs=1` to keep runtime short.
