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
        )

        return loss


# class ACRPN2(Optimizer):
#     def __init__(self, params, eps=1e-8, alpha=required, sigma=1e-3, timesteps=10, eta=1e-3, maximize=False):
#         if alpha is not required and alpha < 0.0:
#             raise ValueError("Invalid alpha parameter: {}".format(alpha))
#         defaults = dict(eps=eps, alpha=alpha, sigma=sigma, timesteps=timesteps, eta=eta,
#                         maximize=maximize)
#         super().__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('maximize', False)
#             # group.setdefault('differentiable', False)
#
#     # @calculate_time
#     def step(self, l1, l2, closure=None):
#
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         group = self.param_groups[0]
#
#         # Optimization logic here
#         acrpn2(
#             params=tuple(group['params']),
#             l1=l1,
#             l2=l2,
#             eps=group['eps'],
#             alpha=group['alpha'],
#             sigma=group['sigma'],
#             timesteps=int(group['timesteps']),
#             eta=group['eta'],
#             maximize=group['maximize'],
#         )
#
#         return loss


# class SGD2(Optimizer):
#     def __init__(self, params, lr=required):
#         if lr is not required and lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         defaults = dict(lr=lr)
#         super().__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('maximize', False)
#             # group.setdefault('differentiable', False)
#
#     # @calculate_time
#     def step(self, l1, closure=None):
#
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         group = self.param_groups[0]
#
#         # Optimization logic here
#         sgd2(
#             params=tuple(group['params']),
#             l1=l1,
#             lr=group['lr'],
#         )
#
#         return loss


# def sgd2(params, l1, lr):
#     grad_estimates = _estimate_grad(params, l1, differentiable=True)
#
#     # # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_forward_jvp()`
#     # _u = torch.ones_like(l2, requires_grad=True)
#     # _g = torch.autograd.grad(l2, params, grad_outputs=_u, create_graph=True)
#     #
#     # # Takes in a vector `v` and calculates the Hessian-vector product
#     # hvp_func = lambda v: _estimate_hvp(params, v, l1, g=_g, u=_u, grad_estimates=grad_estimates)
#     #
#     # delta, val = cubic_subsolver(params, grad_estimates, hvp_func, eps, alpha, sigma, timesteps, eta, maximize)
#     #
#     # if val > -1 / 100 * math.sqrt(eps ** 3 / alpha):
#     #     delta = cubic_finalsolver(params, grad_estimates, hvp_func, eps, alpha, sigma, timesteps, eta, maximize,
#     #                               delta0=delta)
#
#     # Update params
#     grad_norm = _compute_norm(grad_estimates)
#     for i, param in enumerate(params):
#         param.data.add_(grad_estimates[i] / math.sqrt(grad_norm), alpha=-lr)
#         # param.data.add_(grad_estimates[i], alpha=-lr)
#         # param.data.add_(delta[i], alpha=1.)
#
#     # print(_compute_norm(params))


# @torch.jit.script
def acrpn(params: List[Tensor],
          l1: Tensor,
          l2: Tensor,
          eps: float,
          alpha: float,
          sigma: float,
          timesteps: int,
          eta: float):

    grad_estimates = torch.autograd.grad(l1.mean(), params, create_graph=True)
    grad_estimates_detached = list(g.detach() for g in grad_estimates)

    # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
    u = [torch.ones_like(l2, requires_grad=True)]
    uJp = torch.autograd.grad(l2, params, grad_outputs=u, create_graph=True)

    # Takes in a vector `v` and calculates the Hessian-vector product
    hvp_func = lambda v: _estimate_hvp(params, v, l1, uJp=uJp, u=u, grad_estimates=grad_estimates)

    delta, delta_j = cubic_subsolver(params, grad_estimates_detached, hvp_func, alpha, sigma, timesteps, eta)

    if delta_j > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(params, grad_estimates_detached, hvp_func, alpha, eps, timesteps, eta, delta=delta)

    # Update params
    # print(_compute_norm(delta))
    with torch.no_grad():
        for i, param in enumerate(params):
            param.add_(delta[i], alpha=1.)
        # torch._foreach_add_(params, delta)


# def acrpn2(params, l1, l2, eps, alpha, sigma, timesteps, eta, maximize):
#     grad_estimates = _estimate_grad(params, l1, differentiable=True)
#
#     # pre-compute vector-Jacobian product to calculate forward Jacobian-vector product in `_estimate_hvp()`
#     u = torch.ones_like(l2, requires_grad=True)
#     uJp = torch.autograd.grad(l2, params, grad_outputs=u, create_graph=True)
#
#     # Takes in a vector `v` and calculates the Hessian-vector product
#     hvp_func = lambda v: _estimate_hvp(params, v, l1, uJp=uJp, u=u, grad_estimates=grad_estimates)
#
#     delta, val = cubic_subsolver(params, grad_estimates, hvp_func, alpha, sigma, timesteps, eta)
#
#     if val > -1 / 100 * math.sqrt(eps ** 3 / alpha):
#         delta = cubic_finalsolver(params, grad_estimates, hvp_func, alpha, eps, timesteps, eta, delta0=delta)
#
#     # Update params
#     delta_norm = _compute_norm(delta)
#     for i, param in enumerate(params):
#         # param.data.add_(grad_estimates[i], alpha=-1.)
#         param.data.add_(delta[i] / delta_norm, alpha=STEP_SIZE)
#
#     # print(_compute_norm(params))


# @calculate_time
def cubic_subsolver(params, grad_estimates_detached, hvp_func, alpha, sigma, timesteps, eta):
    """
    Implementation of the cubic-subsolver regime as described in Algorithm 2 of Tripuraneni et. al.

    ** Cauchy step **
    R_c  <-  -beta + sqrt( beta^2 + 2*||g|| / alpha), where beta = <g, Hg> / (alpha * ||g||^2)
    Delta  <-  - R_c g / ||g||

    ** Gradient Descent **
     TODO: Finish description

    """

    grad_norm = _compute_norm(grad_estimates_detached)
    print(grad_norm)
    if grad_norm > 1 / alpha:
        # Take Cauchy-Step
        beta = _compute_dot_product(grad_estimates_detached, hvp_func(grad_estimates_detached))
        beta /= alpha * grad_norm * grad_norm
        R_c = -beta + math.sqrt(beta * beta + 2 * grad_norm / alpha)

        delta = list(-R_c * g_detached / grad_norm for g_detached in grad_estimates_detached)
        # print(R_c, R_c / grad_norm, (alpha * beta) ** 2)

    else:
        print('subsolver descent reached')
        perturb = list(torch.randn(g_detach.shape, device=g_detach.device) for g_detach in grad_estimates_detached)
        perturb_norm = _compute_norm(perturb) + 1e-9

        grad_noise = list(g_detach + sigma * grad_norm * per_ / perturb_norm for g_detach, per_ in
                          zip(grad_estimates_detached, perturb))
        # grad_noise_norm = _compute_norm(grad_noise)
        #
        # # Take Cauchy-Step with noisy gradient
        # beta = _compute_dot_product(grad_noise, hvp_func(grad_noise))
        # beta /= alpha * grad_noise_norm * grad_noise_norm
        #
        # R_c = -beta + math.sqrt(beta * beta + 2 * grad_noise_norm / alpha)
        # delta = list(-R_c * g_noise / grad_noise_norm for g_noise in grad_noise)
        delta = list(torch.zeros_like(g_detach) for g_detach in grad_estimates_detached)

        for j in range(timesteps):
            hvp_delta = hvp_func(delta)
            norm_delta = _compute_norm(delta)
            for i, (grad_noise_i, p) in enumerate(zip(grad_noise, params)):
                delta[i][:] -= eta * (grad_noise_i + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

    # Compute the function value
    hvp_delta = hvp_func(delta)
    norm_delta = _compute_norm(delta)

    # delta_j = torch.tensor(0., device=grad_estimates_detached[0].device)
    # for i, p in enumerate(params):
    #     delta_j += (grad_estimates_detached[i] * delta[i]).sum() + 1 / 2 * (
    #             delta[i] * hvp_delta[i]).sum()
    # delta_j += alpha / 6 * norm_delta ** 3

    delta_j = (_compute_dot_product(grad_estimates_detached, delta) + 0.5 * _compute_dot_product(delta, hvp_delta)
               + alpha / 6 * norm_delta ** 3)
    return delta, delta_j


# @calculate_time
def cubic_finalsolver(params, grad_estimates_detached, hvp_func, alpha, eps, timesteps, eta, delta):
    print('finalsolver reached')
    # Start from cauchy point: delta = delta
    grad_iterate = deepcopy(grad_estimates_detached)
    for _ in range(timesteps):
        for i, p in enumerate(params):
            delta[i][:] -= eta * grad_iterate[i]

        hvp_delta = hvp_func(delta)
        norm_delta = _compute_norm(delta)
        # norm_delta = math.sqrt(torch.tensor([(d_*d_).sum() for d_ in delta]).sum().item())
        for i, p in enumerate(params):
            grad_iterate[i][:] = grad_estimates_detached[i] + hvp_delta[i] + alpha / 2 * norm_delta * delta[i]

        norm_grad_iterate = _compute_norm(grad_iterate)
        # norm_grad_iterate = math.sqrt(torch.tensor([(g_*g_).sum() for g_ in grad_iterate]).sum().item())
        # print(norm_grad_iterate, end="\r")
        if norm_grad_iterate < eps / 2:
            break
    return delta


# @calculate_time
# def _estimate_grad(params, l1, differentiable) -> List[Optional[torch.Tensor]]:
#     return torch.autograd.grad(l1.mean(), params, create_graph=differentiable)


def _estimate_hvp(params, v, l1, uJp, u, grad_estimates):
    Jup = torch.autograd.grad(uJp, u, grad_outputs=v, retain_graph=True)[0]  # using fwd autodiff trick
    hvp1 = torch.autograd.grad(l1, params, grad_outputs=Jup / len(Jup), retain_graph=True)

    val = sum((g_ * v_).sum() for g_, v_ in zip(grad_estimates, v))
    hvp2 = torch.autograd.grad(val, params, retain_graph=True)

    return list(h1 + h2 for h1, h2 in zip(hvp1, hvp2))


# @torch.jit.script
def _compute_dot_product(a: List[Optional[torch.Tensor]], b: List[Optional[torch.Tensor]]) -> float:
    # assert len(a) == len(b)
    return torch.tensor([(a_.detach() * b_.detach()).sum() for a_, b_ in zip(a, b)]).sum()


# @torch.jit.script
def _compute_norm(a: List[Optional[torch.Tensor]]) -> float:
    # return math.sqrt(_compute_dot_product(a, a))
    # return torch.norm(torch.cat([a_.ravel() for a_ in a]))
    return math.sqrt((torch.tensor([(a_*a_).sum() for a_ in a])).sum())

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
