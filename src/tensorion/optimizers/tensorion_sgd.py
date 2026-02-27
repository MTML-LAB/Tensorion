import torch

from tensorion import projectors
from functools import partial


class TensorionSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        muon_params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=False,
        nuclear=False,
        adjust_lr=True,
        sgd_params=None,
        sgd_momentum=0.95,
        min_dim: int = 3,
        use_cans_svd: bool = False,
        cans_svd_kwargs: dict | None = None,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            sgd_momentum=sgd_momentum,
            nuclear=nuclear,
            adjust_lr=adjust_lr,
            use_cans_svd=use_cans_svd,
            cans_svd_kwargs=cans_svd_kwargs,
        )

        muon_params = list(muon_params)
        sgd_params = list(sgd_params) if sgd_params is not None else []
        params = muon_params + sgd_params

        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Tensorion for every parameter in muon_params which is >= min_dim and doesn't look like an embedding or head layer
            assert p.ndim >= min_dim, p.ndim
            self.state[p]["use_muon"] = True

        for p in sgd_params:
            # Do not use Muon for parameters in sgd_params
            self.state[p]["use_muon"] = False

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
            use_cans_svd = group.get("use_cans_svd", False)
            cans_svd_kwargs = group.get("cans_svd_kwargs", None)

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

                if use_cans_svd:
                    projector = partial(
                        projectors.project_spectral_sphere,
                        use_cans_svd=True,
                        cans_svd_kwargs=cans_svd_kwargs,
                    )
                else:
                    projector = projectors.project_spectral_sphere

                u = projectors.single_step_projection(
                    x0=g,
                    projector=projector,
                    num_sets=g.dim(),
                )["projection"]

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
            lr = group["lr"]
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
