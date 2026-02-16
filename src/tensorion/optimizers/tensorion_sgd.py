import torch

from .. import tensor_utils
from .. import projectors
from .. import loggers

from typing import Callable


class TensorionSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        projector: Callable[[torch.Tensor], torch.Tensor],
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=False,
        nuclear=False,
        adjust_lr=True,
        sgd_momentum=0.95,
        min_dim: int = 3,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            sgd_momentum=sgd_momentum,
            nuclear=nuclear,
            adjust_lr=adjust_lr,
        )
        
        params = list(params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in params:
            self.state[p]["use_muon"] = (p.ndim >= min_dim)
        
        self.projector = projector
        print(f"{self.projector=}")

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * (max(A, B) ** 0.5)
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nuclear = group["nuclear"]
            adjust_lr = group["adjust_lr"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = self.projector(g)

                if nuclear:
                    nuclear = (g * u).sum()
                    u.mul_(nuclear)

                if adjust_lr:
                    # scale update
                    param_shape = p.shape
                    param_shape = (min(param_shape), g.numel() // min(param_shape))
                    adjusted_lr = self.adjust_lr_for_muon(lr, param_shape)
                else:
                    adjusted_lr = lr

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       SGD backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            mu = group["sgd_momentum"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(g)

                state["step"] += 1

                if weight_decay:
                    g = g.add(p, alpha=weight_decay)

                if mu:
                    momentum_buf = state["momentum"]
                    momentum_buf.mul_(mu).add_(g)
                    g = momentum_buf

                p.add_(g, alpha=-lr)

        return loss
