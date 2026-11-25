import torch


class NoScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Constant lr scheduler

    """

    def __init__(
        self,
        optimizer,
        **kwargs
    ):

        self.lr_0s = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self):
        return self.lr_0s
