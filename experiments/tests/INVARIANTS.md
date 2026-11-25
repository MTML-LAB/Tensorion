# Unit Test Invariants

## tests/unit/test_checkpointing.py
- `test_save_checkpoint_roundtrip`: checkpoint directory `epoch-1` contains `model.pt`, `optimizer.pt`, and `scheduler.pt`, reloaded parameters match original, and scheduler `last_epoch` is preserved; Why: ensures checkpoint round-trip integrity for resume.

## tests/unit/test_data_preparation.py
- `test_load_from_config_builds_train_and_test_loaders`: `load_from_config` returns train/test datasets and loaders with expected sizes, batch shapes `[B, 3, 32, 32]`, and dtypes (`float32` images, `int64` labels); Why: validates data pipeline wiring and basic batch contract.
- `test_train_dataloader_is_deterministic_with_seed`: fixed `seed` yields identical shuffled label order across loaders; Why: deterministic training is reproducible.
- `test_get_dataloader_requires_dataset`: missing dataset for a split raises `ValueError`; Why: avoids silent misconfiguration.
- `test_get_dataloader_missing_split_config_raises`: requesting a split without config raises `ValueError`; Why: enforces explicit dataloader setup.
- `test_get_datasets_requires_dataset_config`: missing `dataset` config raises `ValueError`; Why: ensures required config keys are validated.

## tests/unit/test_model_resnet.py
- `test_resnet18_forward_shape`: ResNet18 returns logits shaped `[B, num_classes]` for input `[B, 3, 32, 32]`; Why: enforces model/output interface for downstream losses.

## tests/unit/test_training_step.py
- `test_train_epoch_updates_params_and_scheduler`: training returns finite loss, processes all samples, updates parameters/gradients, and advances scheduler when `scheduler_step="step"`; Why: confirms one epoch performs a valid optimization step.
- `test_format_lrs_multiple_groups`: multiple optimizer groups are mapped to `lr/group_{idx}` keys; Why: preserves multi-group LR reporting.

## tests/unit/test_train_loop_logging.py
- `test_train_loop_logs_metrics_and_returns_losses`: returns 1 entry per epoch, losses are finite, and logger receives `test/accuracy` and `train/loss` metrics; Why: ensures training loop outputs and logging are consistent.

## tests/unit/test_tensor_utils.py
- `test_unfold_fold_roundtrip`: `fold(unfold(x, mode), shape, mode)` reproduces `x` for all modes and unfold shape is `[mode_dim, N/mode_dim]`; Why: validates tensor reshape algebra used by utilities.
- `test_get_unfoldings_sigvals_nonnegative`: `get_unfoldings_sigvals` returns one 1D tensor per mode and all singular values are non-negative; Why: confirms SVD-based metrics behave as expected.

## tests/unit/test_data_transforms.py
- `test_convert_to_rgb_calls_convert`: `ConvertToRGB` calls `image.convert("RGB")` and returns its result; Why: ensures image normalization is applied consistently.
- `test_build_split_transform_returns_config_entry`: transform config lookup returns the exact split entry; Why: preserves intended transform pipelines.
- `test_build_split_transform_missing_split_raises`: missing split config raises `ValueError`; Why: avoids silent transform omissions.

## tests/unit/test_experiment_logger.py
- `test_flatten_config_filters_sensitive_keys_and_serializes_lists`: sensitive keys (e.g., `api_key`) are excluded, nested keys are dotted, and lists serialize to JSON strings; Why: prevents secret leakage and ensures stable parameter logging.
- `test_build_experiment_logger_disabled_returns_null`: disabled Comet config returns `NullExperimentLogger` and is falsy; Why: avoids external calls when telemetry is off.

## tests/unit/test_loggers.py
- `test_projectors_metrics_collector_factory_collects_metrics`: metrics callback appends one entry per call for `sigvals_bounds`, `dot`, and `residual`, with history sized by tensor rank; Why: ensures metric history tracking is consistent.
- `test_random_grad_logger_records_gradients`: with `log_proba=1.0`, gradients are captured on CPU and persist through `save_logs`/`torch.load`; Why: validates gradient sampling/log persistence.

## tests/unit/test_datasets_cifar.py
- `test_apply_subset_returns_original_for_none`: `subset_size=None` returns the original dataset object; Why: avoids unnecessary wrapping.
- `test_apply_subset_limits_size`: subset size is capped to the requested limit; Why: supports quick experiments on smaller data.
- `test_apply_subset_rejects_non_positive_size`: non-positive `subset_size` raises `ValueError`; Why: enforces valid subset semantics.

## tests/unit/test_train_evaluation.py
- `test_test_returns_accuracy_and_loss`: evaluation returns finite loss and accuracy `1.0` for constant labels with zero logits; Why: confirms evaluation metric computation is correct.
