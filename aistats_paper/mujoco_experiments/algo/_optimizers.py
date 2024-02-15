import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import Dict, List, Tuple, Optional, Callable
import time
from copy import deepcopy


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
                 eps: float = 1e-2,
                 sigma: float = 1e-6,
                 timesteps: int = 10,
                 eta: float = 1e-4,
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
        )

        return loss


# @torch.jit.script
def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          timesteps: int,
          eta: float):

    grad_estimates: List[Tensor] = torch.autograd.grad(l1.mean(), params, create_graph=True)
    grad_estimates_detached = list(g.detach() for g in grad_estimates)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
    u: List[Tensor] = [torch.ones_like(l2, requires_grad=True)]
    uJp: List[Tensor] = torch.autograd.grad(l2, params, grad_outputs=u, create_graph=True)

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: _estimate_hvp(params, v, l1, uJp=uJp, u=u, grad_estimates=grad_estimates)

    delta, delta_j = cubic_subsolver(params, grad_estimates_detached, hvp_func, alpha, sigma, timesteps, eta)

    if delta_j > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(params, grad_estimates_detached, hvp_func, alpha, eps, timesteps, eta, delta=delta)

    # Update params
    with torch.no_grad():
        torch._foreach_add_(params, delta)


# @calculate_time
def cubic_subsolver(params, grad_estimates_detached, hvp_func, alpha, sigma, timesteps, eta):
    """
    Implementation of the cubic-subsolver regime as described in Algorithm 2 of Tripuraneni et. al.

    ** Cauchy step **
    R_c  <-  -beta + sqrt( beta^2 + 2*||g|| / alpha), where beta = <g, Hg> / (alpha * ||g||^2)
    Delta  <-  - R_c g / ||g||

    ** Gradient Descent **
     TODO

    """

    grad_norm = _compute_norm(grad_estimates_detached)

    if grad_norm > 1 / alpha:
        # Take Cauchy-Step
        beta = _compute_dot_product(grad_estimates_detached, hvp_func(grad_estimates_detached))
        beta /= alpha * grad_norm * grad_norm
        R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

        # delta = list(-R_c * g_detached for g_detached in grad_estimates_detached)
        delta = torch._foreach_mul(grad_estimates_detached, -R_c / grad_norm)
        # print(R_c, R_c / grad_norm, (alpha * beta) ** 2)

    else:
        perturb = list(torch.randn(g_detach.shape, device=g_detach.device) for g_detach in grad_estimates_detached)
        perturb_norm = _compute_norm(perturb) + 1e-9

        grad_noise = torch._foreach_add(grad_estimates_detached, perturb, alpha=grad_norm*sigma/perturb_norm)
        grad_noise_norm = _compute_norm(grad_noise)

        # Take Cauchy-Step with noisy gradient
        beta = _compute_dot_product(grad_noise, hvp_func(grad_noise))
        beta /= alpha * grad_noise_norm * grad_noise_norm

        R_c = -beta + math.sqrt(beta * beta + 2 * grad_noise_norm / alpha)
        delta = torch._foreach_mul(grad_estimates_detached, -R_c / grad_noise_norm)

        for j in range(timesteps):
            hvp_delta = hvp_func(delta)
            norm_delta = _compute_norm(delta)
            for i, grad_noise_i in enumerate(grad_noise):
                delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

    # Compute the function value
    hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    # delta_j = torch.tensor(0., device=grad_estimates_detached[0].device)
    # for i, g_i in enumerate(grad_estimates_detached):
    #     delta_j += (g_i * delta[i]).sum() + 0.5 * (delta[i] * hvp_delta[i]).sum() + alpha / 6 * norm_delta ** 3

    delta_j = (_compute_dot_product(grad_estimates_detached, delta) + 0.5 * _compute_dot_product(delta, hvp_delta)
               + alpha / 6 * norm_delta ** 3)

    return delta, delta_j


# @calculate_time
def cubic_finalsolver(params, grad_estimates_detached, hvp_func, alpha, eps, timesteps, eta, delta):
    # Start from cauchy point: delta = delta
    grad_iterate = deepcopy(grad_estimates_detached)
    for _ in range(timesteps):
        torch._foreach_add_(delta, grad_iterate, alpha=-eta)

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)
        grad_iterate = torch._foreach_add(grad_estimates_detached, hvp_delta)
        torch._foreach_add_(grad_iterate, delta, alpha=alpha / 2 * norm_delta)

        norm_grad_iterate = _compute_norm(grad_iterate)
        # print(norm_grad_iterate, end="\r")
        if norm_grad_iterate < eps / 2:
            break
    return delta


def _estimate_hvp(params, v, l1, uJp, u, grad_estimates):
    Jup = torch.autograd.grad(uJp, u, grad_outputs=v, retain_graph=True)[0]  # using fwd autodiff trick
    hvp1 = torch.autograd.grad(l1, params, grad_outputs=Jup / len(Jup), retain_graph=True)

    val = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))
    hvp2 = torch.autograd.grad(val, params, retain_graph=True)

    torch._foreach_add_(hvp1, hvp2)
    return hvp1


# @torch.jit.script
def _compute_dot_product(a: List[Optional[torch.Tensor]], b: List[Optional[torch.Tensor]]) -> float:
    # assert len(a) == len(b)
    return torch.tensor([ab.sum() for ab in torch._foreach_mul(a, b)], device=a[0].device).sum()


# @torch.jit.script
def _compute_norm(a: List[Optional[torch.Tensor]]) -> float:
    return math.sqrt(torch.tensor([a2.sum() for a2 in torch._foreach_mul(a, a)], device=a[0].device).sum())