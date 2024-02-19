import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
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
                 timesteps: int = 100,
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
    def step(self, l1, l2, closure=None):  # Todo: Try incorporating the loses under closure

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


def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          timesteps: int,
          eta: float):

    grad_estimates = list(torch.autograd.grad(l1.mean(), params, create_graph=True))
    grad_estimates_detached = list(g.detach() for g in grad_estimates)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
    u = [torch.ones_like(l2, requires_grad=True)]
    uJp = list(torch.autograd.grad(l2, params, grad_outputs=u, create_graph=True))

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: _estimate_hvp(params, v, l1, uJp=uJp, u=u, grad_estimates=grad_estimates)

    delta, delta_j = cubic_subsolver(grad_estimates_detached, hvp_func, alpha, sigma, timesteps, eta)

    if delta_j > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(grad_estimates_detached, hvp_func, alpha, eps, timesteps, eta, delta=delta)

    # Update params
    with torch.no_grad():
        torch._foreach_add_(params, delta)


def cubic_subsolver(grad_estimates_detached: List[Tensor],
                    hvp_func: Callable[[List[Tensor]], List[Tensor]],
                    alpha: float,
                    sigma: float,
                    timesteps: int,
                    eta: float):
    """
    Implementation of the cubic-subsolver regime as described in Algorithm 2 of Tripuraneni et. al.

    ** Cauchy step **
    R_c  <-  -beta + sqrt( beta^2 + 2*||g|| / alpha), where beta = <g, Hg> / (alpha * ||g||^2)
    Delta  <-  - R_c g / ||g||

    ** Gradient Descent **
     TODO: Finish description

    """

    grad_norm = _compute_norm(grad_estimates_detached)

    if grad_norm > 1 / alpha:
        # Take Cauchy-Step
        beta = _compute_dot_product(grad_estimates_detached, hvp_func(grad_estimates_detached))
        beta /= alpha * grad_norm * grad_norm
        R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

        # Cauchy point
        delta = torch._foreach_mul(grad_estimates_detached, -R_c / grad_norm)

    else:
        perturb = list(torch.randn(g_detach.shape, device=g_detach.device) for g_detach in grad_estimates_detached)
        perturb_norm = _compute_norm(perturb) + 1e-9

        grad_noise = torch._foreach_add(grad_estimates_detached, perturb, alpha=grad_norm*sigma/perturb_norm)

        # Todo: Figure out whether or not to implement below commented code
        # grad_noise_norm = _compute_norm(grad_noise)
        #
        # # Take Cauchy-Step with noisy gradient
        # beta = _compute_dot_product(grad_noise, hvp_func(grad_noise))
        # beta /= alpha * grad_noise_norm * grad_noise_norm
        #
        # R_c = -beta + math.sqrt(beta * beta + 2 * grad_noise_norm / alpha)
        # delta = torch._foreach_mul(grad_estimates_detached, -R_c / grad_noise_norm)

        delta = list(torch.zeros_like(g_detach) for g_detach in grad_estimates_detached)

        for j in range(timesteps):
            hvp_delta = hvp_func(delta)
            norm_delta = _compute_norm(delta)
            for i, grad_noise_i in enumerate(grad_noise):
                delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

    # Compute the function value
    hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    delta_j = (_compute_dot_product(grad_estimates_detached, delta) + 0.5 * _compute_dot_product(delta, hvp_delta)
               + alpha / 6 * norm_delta ** 3)

    return delta, delta_j


def cubic_finalsolver(grad_estimates_detached: List[Tensor],
                      hvp_func: Callable[[List[Tensor]], List[Tensor]],
                      alpha: float,
                      eps: float,
                      timesteps: int,
                      eta: float,
                      delta: List[Tensor]):
    """
    TODO : Add description
    """

    # Start from cauchy point, i.e. delta = delta
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


def _estimate_hvp(params: List[Tensor],
                  v: List[Tensor],
                  l1: Tensor,
                  uJp: List[Tensor],
                  u: List[Tensor],
                  grad_estimates: List[Optional[Tensor]]):

    Jvp = torch.autograd.grad(uJp, u, grad_outputs=v, retain_graph=True)[0]  # using fwd autodiff trick
    hvp1 = list(torch.autograd.grad(l1, params, grad_outputs=Jvp / len(Jvp), retain_graph=True))

    # TODO: Try using ``_compute_dot_product`` here instead
    gTv = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))  # inner product <grad, v>
    hvp2 = list(torch.autograd.grad(gTv, params, retain_graph=True))

    torch._foreach_add_(hvp1, hvp2)
    return hvp1


def _compute_dot_product(a: List[Optional[torch.Tensor]], b: List[Optional[torch.Tensor]]) -> float:
    return torch.tensor([ab.sum() for ab in torch._foreach_mul(a, b)], device=a[0].device).sum()


def _compute_norm(a: List[Optional[torch.Tensor]]) -> float:
    return math.sqrt(torch.tensor([a2.sum() for a2 in torch._foreach_mul(a, a)], device=a[0].device).sum())
