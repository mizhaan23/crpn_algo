import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import Dict, List, Tuple, Optional, Callable
import time


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to execute {func.__name__}: {elapsed_time} seconds")
        return result

    return wrapper


class ACRPN(Optimizer):
    def __init__(self,
                 params: List[Tensor],
                 alpha: float,
                 eps: float = 1e-8,
                 sigma: float = 1e-3,
                 timesteps: int = 10,
                 eta: float = 1e-3,
                 maximize: bool = False):

        if alpha < 0.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        defaults = dict(eps=eps, alpha=alpha, sigma=sigma, timesteps=timesteps, eta=eta,
                        maximize=maximize)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            # group.setdefault('differentiable', False)

    # @calculate_time
    def step(self, l1, l2, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        # Optimization logic here
        acrpn(
            params=list(group['params']),
            l1=l1,
            l2=l2,
            eps=group['eps'],
            alpha=group['alpha'],
            sigma=group['sigma'],
            timesteps=int(group['timesteps']),
            eta=group['eta'],
            maximize=group['maximize'],
        )

        return loss


# @torch.jit.script
# class FunctionalACRPN(object):
#
# def __init__(self,
#              params: List[Tensor],
#              alpha: float,
#              eps: float = 1e-8,
#              sigma: float = 1e-3,
#              timesteps: int = 10,
#              eta: float = 1e-3,
#              maximize: bool = False):
#         if alpha < 0.0:
#             raise ValueError(f"Invalid alpha parameter: {alpha}")
#
#         self.defaults = dict(alpha=alpha, eps=eps, sigma=sigma, timesteps=timesteps, eta=eta,
#                              maximize=maximize)
#
#         self.param_group = {"params": params}
#         self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
#
#     def step(self, l1, l2, closure=None):
#
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         # group = self.param_groups[0]
#
#         params_with_grad = []
#         for param in self.param_group['params']:
#             params_with_grad.append(param)
#
#         # Optimization logic here
#         acrpn(
#             params=params_with_grad,
#             l1=l1,
#             l2=l2,
#             eps=self.defaults['eps'],
#             alpha=self.defaults['alpha'],
#             sigma=self.defaults['sigma'],
#             timesteps=self.defaults['timesteps'],
#             eta=self.defaults['eta'],
#             maximize=self.defaults['maximize'],
#         )
#
#         return loss


# @torch.jit.script
def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          timesteps: int,
          eta: float,
          maximize: bool):

    grad_estimates: List[Tensor] = _estimate_grad(params, l1, differentiable=True)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_forward_jvp()`
    _u: List[Tensor] = [torch.ones_like(l2, requires_grad=True)]
    # _g: List[Optional[Tensor]] = torch.autograd.grad(l2, params, grad_outputs=_u, create_graph=True)
    _g: List[Tensor] = script_autograd([l2,], params, grad_outputs=_u, create_graph=True)

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: _estimate_hvp(params, v, l1, g=_g, u=_u, grad_estimates=grad_estimates)

    # temp = _estimate_hvp(params, v=grad_estimates, l1=l1, g=_g, u=_u, grad_estimates=grad_estimates)

    delta, val = cubic_subsolver(params=params, grad_estimates=grad_estimates, eps=eps, alpha=alpha, sigma=sigma,
                                 timesteps=timesteps, eta=eta, maximize=maximize, l1=l1, g=_g, u=_u)

    if val > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(params, grad_estimates, hvp_func, eps, alpha, sigma, timesteps, eta, maximize,
                                  delta0=delta)

    # Update params
    for i, param in enumerate(params):
        # param.data.add_(grad_estimates[i], alpha=-1.)
        param.data.add_(delta[i], alpha=1.)

    # print(_compute_norm(params))


# @calculate_time

# @calculate_time
def cubic_finalsolver(params, grad_estimates, hvp_func, eps, alpha, sigma, timesteps, eta, maximize, delta0=None):
    if delta0 is None:
        delta = list(torch.zeros_like(g_) for g_ in grad_estimates)

    else:  # Start from the Cauchy Point
        delta = list(d_.detach().clone() for d_ in delta0)

    # grad_iterate = [0.] * len(grad_estimates)
    grad_iterate = list(g_.detach().clone() for g_ in grad_estimates)  # [0.] * len(grad_estimates)

    for _ in range(timesteps):
        for i, p in enumerate(params):
            delta[i][:] -= eta * grad_iterate[i]

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)
        for i, p in enumerate(params):
            grad_iterate[i][:] = grad_estimates[i].detach() + hvp_delta[i] + alpha / 2 * norm_delta * delta[i]

        norm_grad_iterate = _compute_norm(grad_iterate)
        # print(norm_grad_iterate)
        if norm_grad_iterate < eps / 2:
            break

    return delta


# @calculate_time

@torch.jit.script
def script_autograd(outputs: List[Tensor],
                    inputs: List[Tensor],
                    grad_outputs: List[Tensor],
                    retain_graph: bool = False,
                    create_graph: bool = False,
                    ) -> List[Tensor]:

    if create_graph:
        retain_graph = True

    grad_outputs: List[Optional[Tensor]] = [k for k in grad_outputs]
    grad = torch.autograd.grad(outputs=outputs,
                               inputs=inputs,
                               grad_outputs=grad_outputs,
                               retain_graph=retain_graph,
                               create_graph=create_graph)

    res = torch.jit.annotate(List[Tensor], [])
    for i, g in enumerate(grad):
        if g is not None:
            res.append(g)
        else:
            print('here')
            res.append(torch.zeros_like(inputs[i]))
    return res


@torch.jit.script
def _estimate_grad(params: List[Tensor], l1: Tensor, differentiable: bool) -> List[Tensor]:
    # return torch.autograd.grad([l1.mean()], params, create_graph=differentiable)
    grad_outputs: List[Tensor] = [torch.ones_like(l1) / len(l1)]
    # return torch.autograd.grad([l1,], params, grad_outputs=grad_outputs, create_graph=differentiable)
    out = script_autograd([l1,], params, grad_outputs, create_graph=differentiable)
    return out


@torch.jit.script
def _forward_jvp(u: List[Tensor],
                 g: List[Tensor],
                 v: List[Tensor]) -> List[Tensor]:
    return script_autograd(g, u, grad_outputs=v, retain_graph=True)


@torch.jit.script
def _estimate_hvp(params: List[Tensor],
                  v: List[Tensor],
                  l1: Tensor,
                  g: List[Tensor],
                  u: List[Tensor],
                  grad_estimates: List[Tensor]) -> List[Tensor]:

    # Detach v from grad graph
    v = list(v_.detach() for v_ in v)

    w = _forward_jvp(u, g, v)[0]  # using fwd autodiff trick

    val: Tensor = torch.tensor(0., device=grad_estimates[0].device)
    for g_, v_ in zip(grad_estimates, v):
        val += (g_ * v_).sum()

    hvp1 = script_autograd([l1,], params, grad_outputs=[w / len(w),], retain_graph=True)
    hvp2 = script_autograd([val,], params, grad_outputs=[torch.ones_like(val),], retain_graph=True)

    return [h1+h2 for h1, h2 in zip(hvp1, hvp2)]


@torch.jit.script
def _compute_dot_product(a: List[Tensor], b: List[Tensor]) -> Tensor:
    assert len(a) == len(b)
    res: Tensor = torch.tensor(0., device=a[0].device)
    for a_, b_ in zip(a, b):
        res += (a_.detach() * b_.detach()).sum()
    return res


@torch.jit.script
def _compute_norm(a: List[Tensor]) -> Tensor:
    return torch.sqrt(_compute_dot_product(a, a))


@torch.jit.script
def cubic_subsolver(params: List[Tensor],
                    grad_estimates: List[Tensor],
                    eps: float,
                    alpha: float,
                    sigma: float,
                    timesteps: int,
                    eta: float,
                    maximize: bool,
                    l1: Tensor,
                    g: List[Tensor],
                    u: List[Tensor],
                    ) -> Tuple[List[Tensor], Tensor]:

    grad_norm = _compute_norm(grad_estimates)

    if grad_norm > 1 / alpha:
        hvp_grad = _estimate_hvp(params=params, v=grad_estimates, l1=l1, g=g, u=u, grad_estimates=grad_estimates)
        # hvp_grad = hvp_func(grad_estimates)
        beta = _compute_dot_product(grad_estimates, hvp_grad)
        beta /= alpha * grad_norm * grad_norm
        R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

        delta = list(-R_c * g_.detach() / grad_norm for g_ in grad_estimates)
        # print(R_c)

    else:
        # print('here')
        delta = list(torch.zeros_like(g_) for g_ in grad_estimates)
        perturb = list(torch.randn(g_.shape, device=g_.device) for g_ in grad_estimates)
        perturb_norm = _compute_norm(perturb) + 1e-9

        grad_noise = (g_.detach() + sigma * per_ / perturb_norm for g_, per_ in zip(grad_estimates, perturb))

        for j in range(timesteps):
            hvp_delta = _estimate_hvp(params=params, v=delta, l1=l1, g=g, u=u, grad_estimates=grad_estimates)
            # hvp_delta = hvp_func(delta)
            norm_delta = _compute_norm(delta)
            for i, (grad_noise_i, p) in enumerate(zip(grad_noise, params)):
                delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

    # Compute the function value
    hvp_delta = _estimate_hvp(params=params, v=delta, l1=l1, g=g, u=u, grad_estimates=grad_estimates)
    # hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    val: Tensor = torch.tensor(0., device=grad_estimates[0].device)
    for i, p in enumerate(params):
        val += (grad_estimates[i] * delta[i]).sum() + 1 / 2 * (
                delta[i] * hvp_delta[i]).sum() + alpha / 6 * norm_delta ** 3

    return delta, val


# @calculate_time
# @torch.jit.script  # OPTIMIZE!!!
# def _pre_compute_l2_grad(params: List[torch.Tensor], l2) -> List[List[Optional[torch.Tensor]]]:
#     return [torch.autograd.grad([l_,], params, retain_graph=True) for l_ in l2]11

# def _forward_jvp(params, l2, u):
#     v = torch.ones_like(l2, requires_grad=True)
#     g = torch.autograd.grad(l2, params, grad_outputs=v, create_graph=True)  # CACHE THIS
#     return torch.autograd.grad(g, v, grad_outputs=u, retain_graph=False)


# def _estimate_hvp(params, l1, v, l2_grad, grad_estimates=None):
#     assert (grad_estimates is not None)
#
#     # Detach v from grad graph
#     v = tuple(v_.detach() for v_ in v)
#
#     w = torch.stack([(sum((t_ * v_).sum() for t_, v_ in zip(t, v))) for t in l2_grad])  # of size (batch_size)
#     val = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))
#
#     Hvp1_torch = torch.autograd.grad(l1, params, grad_outputs=w / len(w), retain_graph=True)
#     Hvp2_torch = torch.autograd.grad(val, params, retain_graph=True)
#
#     Hvp_torch = tuple(h1 + h2 for h1, h2 in zip(Hvp1_torch, Hvp2_torch))
#     return Hvp_torch
#
#
# def _estimate_hvp2(params, l1, l2, v, grad_estimates=None):
#     assert (grad_estimates is not None)
#
#     # Detach v from grad graph
#     v = tuple(v_.detach() for v_ in v)
#
#     w = _forward_jvp(params, l2, v)[0]  # using trick
#     val = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))
#
#     Hvp1_torch = torch.autograd.grad(l1, params, grad_outputs=w / len(w), retain_graph=True)
#     Hvp2_torch = torch.autograd.grad(val, params, retain_graph=True)
#
#     Hvp_torch = tuple(h1 + h2 for h1, h2 in zip(Hvp1_torch, Hvp2_torch))
#     return Hvp_torch
