# Tensorion optimizer for matrix and tensor layers

PyTorch optimizer library for matrix and tensor layers. Install just the optimizer core, or add optional extras for experiments and validation.

## Installation

### Core (optimizers only)

```bash
pip install tensorion
# or from source
uv pip install . --no-build-isolation
```

Core dependencies: `numpy`, `torch`, `scipy`

### Extras

- **`experiments`** - Configuration, logging, training pipelines  
  ```bash
  pip install "tensorion[experiments]"
  ```

- **`cv`** - Vision models and HuggingFace integration  
  ```bash
  pip install "tensorion[cv]"
  ```

- **`optimization`** - Additional optimization backends (e.g., `cvxpy`)  
  ```bash
  pip install "tensorion[optimization]"
  ```

- **`dev`** - Testing and development tools  
  ```bash
  pip install "tensorion[dev]"
  ```

Combine extras as needed:
```bash
pip install "tensorion[experiments,cv,optimization]"
```

## Project Structure

**Core (always installed):**
- `src/tensorion/optimizers/` - Optimizer implementations
- `src/tensorion/tensor_utils.py`, `projectors.py` - Core utilities

**Experiments & validation (optional extras):**
- `src/tensorion/datasets/`, `models/`, `train.py`, `run_experiment.py`
- `src/tensorion/loggers.py`, `experiment_logger.py`, `eval.py`
- `configs/` - Experiment configurations

## Usage

```python
import torch
from tensorion.optimizers import Tensorion

model = model(...)
optimizer = Tensorion(
    muon_params=[p for p in model.parameters() if p.ndim >= 3],
    adamw_params=[p for p in model.parameters() if p.ndim < 3],
    lr=0.01,
    wd=0.1,
    momentum=0.95,
)
```
