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
                 sigma: float = 1e-3,
                 maxiter: int = 100,
                 eta: float = 1e-4,
                 maximize: bool = False):

        if alpha < 0.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if maxiter < 1:
            raise ValueError(f"maxiter should be greater than or equal to 1 and not {maxiter}")

        defaults = dict(eps=eps, alpha=alpha, sigma=sigma, maxiter=maxiter, eta=eta,
                        maximize=maximize)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

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
            maxiter=int(group['maxiter']),
            eta=group['eta'],
        )

        return loss


def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          maxiter: int,
          eta: float):

    grad_estimates = list(torch.autograd.grad(l1.mean(), params, create_graph=True))
    grad_estimates_detached = list(g.detach() for g in grad_estimates)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
    u = [torch.ones_like(l2, requires_grad=True)]
    uJp = list(torch.autograd.grad(l2, params, grad_outputs=u, create_graph=True))

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: _estimate_hvp(params, v, l1, uJp=uJp, u=u, grad_estimates=grad_estimates)

    delta, f_delta = cubic_subsolver(grad_estimates_detached, hvp_func, alpha, eps, sigma, maxiter, eta)

    if f_delta > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(grad_estimates_detached, hvp_func, alpha, eps, maxiter, eta, delta=delta)

    # Update params
    with torch.no_grad():
        torch._foreach_add_(params, delta)


def cubic_subsolver(grad_estimates_detached: List[Tensor],
                    hvp_func: Callable[[List[Tensor]], List[Tensor]],
                    alpha: float,
                    eps: float,
                    sigma: float,
                    maxiter: int,
                    eta: float):
    """
    Implementation of the cubic-subsolver regime as described in Carmon & Duchi paper

    define: f(x; g) = 1/2 <x, H[x]> + <g, x> + alpha / 6 ||x||^3 [Cubic sub-model]
            f'(x; g) = H[x] + g + alpha / 2 ||x|| x
            threshold = -1 / 100 * eps^(3/2) * alpha^(-1/2)

    ** Cauchy step **
    R_c  <-  -beta + sqrt( beta^2 + 2*||g|| / alpha), where beta = <g, Hg> / (alpha * ||g||^2)
    Delta  <-  -R_c * g / ||g||

    if f(Delta) < threshold:
        return Delta, f(Delta)
    else:
        GO TO GRADIENT DESCENT

    ** Gradient Descent **

    g_noise <- g + sigma * q, where q ~ Uniformly from a sphere

    ** Cauchy step with noise **
    R_c_noise  <-  -beta_noise + sqrt( beta_noise^2 + 2*||g_noise|| / alpha),
    where beta_noise = <g_noise, H g_noise> / (alpha * ||g_noise||^2)

    Delta_0  <-  -R_c_noise * g_noise / ||g_noise||

    for 1, 2, ..., maxiter:
        Delta_t <- Delta_{t-1} - eta * (g_noise + H[g_noise] + alpha / 2 * ||Delta_{t-1}|| * Delta_{t-1})

        if f(Delta_t; g) < threshold:
            return Delta_t, f(Delta_t; g)

    return Delta_t, f(Delta_t)
    """

    # Take Cauchy-Step
    grad_norm = _compute_norm(grad_estimates_detached)

    beta = _compute_dot_product(grad_estimates_detached, hvp_func(grad_estimates_detached))
    beta /= alpha * grad_norm * grad_norm
    R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

    # Cauchy point
    delta = torch._foreach_mul(grad_estimates_detached, -R_c / grad_norm)

    # sub-model value at delta_0 (Cauchy-point), i.e. f(delta_0)
    # where f(x) = 1/2 <x, H[x]> + <g, x> + alpha / 6 ||x||^3 [Cubic sub-model]
    # -> f(delta_0) = - 1/2 * (R_c ||g|| + alpha / 6 * R_c^3)
    f_delta = - 1 / 2 * (R_c * grad_norm + alpha / 6 * R_c ** 3)

    if f_delta < -1 / 100 * math.sqrt(eps ** 3 / alpha):
        return delta, f_delta

    # If above condition is not satisfied, try noisy gradient descent
    q = list(torch.randn(g_detach.shape, device=g_detach.device) for g_detach in grad_estimates_detached)
    q_norm = _compute_norm(q) + 1e-9

    grad_noise = torch._foreach_add(grad_estimates_detached, q, alpha=sigma/q_norm)

    # Take Cauchy-Step with noisy gradient
    grad_noise_norm = _compute_norm(grad_noise)

    beta_noise = _compute_dot_product(grad_noise, hvp_func(grad_noise))
    beta_noise /= alpha * grad_noise_norm * grad_noise_norm
    R_c_noise = -beta_noise + math.sqrt(beta_noise * beta_noise + 2 * grad_noise_norm / alpha)

    # Cauchy point w noisy grad
    delta = torch._foreach_mul(grad_estimates_detached, -R_c_noise / grad_noise_norm)

    hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    for _ in range(maxiter):
        # Update delta in-place
        for i, grad_noise_i in enumerate(grad_noise):
            delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)

        # sub-model value at delta
        f_delta = (0.5 * _compute_dot_product(delta, hvp_delta) +
                   _compute_dot_product(grad_estimates_detached, delta) +
                   alpha / 6 * norm_delta ** 3)

        # If condition is reached, break early
        if f_delta < -1 / 100 * math.sqrt(eps ** 3 / alpha):
            return delta, f_delta

    return delta, f_delta


def cubic_finalsolver(grad_estimates_detached: List[Tensor],
                      hvp_func: Callable[[List[Tensor]], List[Tensor]],
                      alpha: float,
                      eps: float,
                      maxiter: int,
                      eta: float,
                      delta: List[Tensor]):
    """
    TODO : Add description
    """

    # Start from cauchy point, i.e. delta = delta
    grad_iterate = deepcopy(grad_estimates_detached)
    for _ in range(maxiter):
        torch._foreach_add_(delta, grad_iterate, alpha=-eta)

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)
        grad_iterate = torch._foreach_add(grad_estimates_detached, hvp_delta)
        torch._foreach_add_(grad_iterate, delta, alpha=alpha / 2 * norm_delta)

        norm_grad_iterate = _compute_norm(grad_iterate)
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
