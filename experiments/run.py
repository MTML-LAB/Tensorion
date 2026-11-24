import sys

import hydra
from omegaconf import DictConfig

from tensorion_experiments import run_experiment
from tensorion_experiments.config_resolvers import register_resolvers

register_resolvers()

@hydra.main(config_path="./configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    run_experiment.run(cfg)


if __name__ == "__main__":
    main()
