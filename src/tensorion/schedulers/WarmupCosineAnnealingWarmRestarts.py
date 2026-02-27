import torch
import math


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """
    Cosine Annealing with Warmup and Warm Restarts.

    Behavior:
    1) For the first T_warmup steps, the learning rate is increased linearly
       from lr_0 to base_lr (lr_max) for each parameter group.
    2) After warmup, the scheduler behaves like CosineAnnealingWarmRestarts:
       cosine cycles of length T_0, then T_0 * T_mult, T_0 * T_mult^2, etc.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_warmup (int): Number of warmup steps.
        T_0 (int): Number of steps in the first cosine cycle.
        T_mult (int, optional): Multiplicative factor for cycle length growth.
            If 1, all cycles have the same length. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.0.
        lr_0 (float | None, optional): Initial learning rate at step 0.
            If None, the current LR in optimizer is used as lr_0. Default: 0.0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        T_warmup: int,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        lr_0: float | None = 0.0,
        last_epoch: int = -1,
    ):
        if T_warmup < 0:
            raise ValueError("T_warmup must be >= 0")
        if T_0 <= 0:
            raise ValueError("T_0 must be > 0")
        if T_mult < 1:
            raise ValueError("T_mult must be >= 1")

        self.T_warmup = T_warmup
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        if lr_0 is None:
            # disabled warmup
            self.lr_0s = [group["lr"] for group in optimizer.param_groups]
        else:
            # enabled warmup (default: lr_0 = 0.0)
            self.lr_0s = [float(lr_0) for _ in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup
        if (self.last_epoch < self.T_warmup) and (self.T_warmup > 0):
            warmup_progress = (self.last_epoch + 1) / self.T_warmup

            return [
                lr_0 + (base_lr - lr_0) * warmup_progress
                for lr_0, base_lr in zip(self.lr_0s, self.base_lrs)
            ]

        # Cosine Annealing with warm restarts
        t = self.last_epoch - self.T_warmup
        return self._get_cosine_lrs(t)

    def _get_cosine_lrs(self, t: int):
        """
        Compute cosine-annealed learning rates for a given step index t in the
        cosine phase (after warmup).

        Args:
            t (int): Step index starting from 0 after the warmup phase.

        Returns:
            list[float]: List of learning rates for each parameter group.
        """

        T_i = self.T_0
        T_cur = t
        if self.T_mult == 1:
            T_cur = t % T_i
        else:
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult

        # cosine-annealing
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * T_cur / T_i))
        return [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]
