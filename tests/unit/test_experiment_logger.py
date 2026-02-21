from omegaconf import OmegaConf

from tensorion import experiment_logger


def test_flatten_config_filters_sensitive_keys_and_serializes_lists():
    cfg = OmegaConf.create(
        {
            "comet": {"api_key": "secret", "project": "demo"},
            "dataset": {"classes": [1, 2, 3]},
            "nested": {"value": 7},
        }
    )

    flattened = experiment_logger.flatten_config(cfg)

    assert "comet.api_key" not in flattened
    assert flattened["comet.project"] == "demo"
    assert flattened["dataset.classes"] == "[1, 2, 3]"
    assert flattened["nested.value"] == 7


def test_build_experiment_logger_disabled_returns_null():
    cfg = OmegaConf.create({"comet": {"enabled": False}})

    logger = experiment_logger.build_experiment_logger(cfg)

    assert isinstance(logger, experiment_logger.NullExperimentLogger)
    assert not logger
