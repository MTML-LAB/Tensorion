import torch


def _constant_lr(_step: int) -> float:
    return 1.0


class NoOpScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    No-op scheduler that keeps the learning rate constant.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, lr_lambda=_constant_lr, last_epoch=last_epoch)

    def step(self, metrics: object | None = None, **kwargs: object) -> None:
        # Ignore metrics/epoch arguments to preserve a constant LR.
        super().step()
