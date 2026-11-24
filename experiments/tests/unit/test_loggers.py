import torch
from torch import nn

from tensorion_experiments import loggers


def test_projectors_metrics_collector_factory_collects_metrics():
    x0 = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    callback, history = loggers.projectors_metrics_collector_factory(
        x0, metrics=["sigvals_bounds", "dot", "residual"]
    )

    callback(x0 + 1, i=0, step=0)

    assert len(history["max_sigvals"]) == x0.dim()
    assert len(history["min_sigvals"]) == x0.dim()
    for idx in range(x0.dim()):
        assert len(history["max_sigvals"][idx]) == 1
        assert len(history["min_sigvals"][idx]) == 1
    assert len(history["dot"]) == 1
    assert len(history["res_norm"]) == 1


def test_random_grad_logger_records_gradients(tmp_path):
    model = nn.Linear(4, 2)
    inputs = torch.randn(3, 4)
    targets = torch.zeros(3, dtype=torch.long)

    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()

    logger = loggers.RandomGradLogger(model, log_proba=1.0, seed=0)
    logger.sample_log(global_step=5)

    assert logger.grad_logs
    step, name, grad = logger.grad_logs[0]
    assert step == 5
    assert isinstance(name, str)
    assert grad.device.type == "cpu"

    out_file = tmp_path / "grads.pt"
    logger.save_logs(str(out_file))

    saved = torch.load(out_file)
    assert len(saved) == len(logger.grad_logs)
    assert saved[0][0] == logger.grad_logs[0][0]
    assert saved[0][1] == logger.grad_logs[0][1]
    assert torch.equal(saved[0][2], logger.grad_logs[0][2])
