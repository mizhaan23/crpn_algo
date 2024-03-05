import math

import torch
from torch.optim.optimizer import Optimizer, required


class ACRPN(Optimizer):
    def __init__(self, params, eps=1e-8, alpha=required, sigma=1e-3, timesteps=100, eta=1e-2, maximize=False):
        if alpha is not required and alpha < 0.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        defaults = dict(eps=eps, alpha=alpha, sigma=sigma, timesteps=timesteps, eta=eta,
                        maximize=maximize)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            # group.setdefault('differentiable', False)

    def step(self, l1, l2, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        # Optimization logic here
        acrpn(
            params=tuple(group['params']),
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


def acrpn(params, l1, l2, eps, alpha, sigma, timesteps, eta, maximize):
    grad_estimates = _estimate_grad(params, l1, differentiable=True)
    Hvp_estimators = lambda v: _estimate_hvp(params, l1, l2, v, grad_estimates=grad_estimates)

    delta, val = cubic_subsolver(params, grad_estimates, Hvp_estimators, eps, alpha, sigma, timesteps, eta, maximize)

    if val > -1 / 100 * math.sqrt(eps ** 3 / alpha):
        delta = cubic_finalsolver(params, grad_estimates, Hvp_estimators, eps, alpha, sigma, timesteps, eta, maximize)

    # Update params
    for i, param in enumerate(params):
        # param.data.add_(grad_estimates[i], alpha=-1.)
        param.data.add_(delta[i], alpha=1.)


def cubic_subsolver(params, grad_estimates, Hvp_estimators, eps, alpha, sigma, timesteps, eta, maximize):
    grad_norm = _compute_norm(grad_estimates)

    if grad_norm > 1 / alpha:
        t = _compute_dot_product(grad_estimates, Hvp_estimators(grad_estimates))
        t = t / (alpha * grad_norm * grad_norm)
        R_c = -t + math.sqrt(t * t + 2 * grad_norm / alpha)

        delta = list(-R_c * g_ / grad_norm for g_ in grad_estimates)

    else:
        delta = list(torch.zeros_like(g_) for g_ in grad_estimates)
        perturb = tuple(torch.randn(g_.shape) for g_ in grad_estimates)
        perturb_norm = _compute_norm(perturb)

        grad_noise = tuple(g_ + sigma * per_ / (perturb_norm + 1e-9) for g_, per_ in zip(grad_estimates, perturb))

        for j in range(timesteps):
            hvp_delta = Hvp_estimators(delta)
            norm_delta = _compute_norm(delta)
            for i, p in enumerate(params):
                delta[i] -= eta * (grad_noise[i] + hvp_delta[i] + alpha / 2 * norm_delta * delta[i])

    # Compute the function value
    hvp_delta = Hvp_estimators(delta)
    norm_delta = _compute_norm(delta)

    val = 0.
    for i, p in enumerate(params):
        val += (grad_estimates[i] * delta[i]).sum() + 1 / 2 * (
                delta[i] * hvp_delta[i]).sum() + alpha / 6 * norm_delta ** 3

    return delta, val


def cubic_finalsolver(params, grad_estimates, Hvp_estimators, eps, alpha, sigma, timesteps, eta, maximize):
    delta = list(torch.zeros_like(g_) for g_ in grad_estimates)
    grad_iterate = list(grad_estimates)

    for j in range(timesteps):
        for i, p in enumerate(params):
            delta[i] -= eta * grad_iterate[i]

        hvp_delta = Hvp_estimators(delta)
        norm_delta = _compute_norm(delta)
        for i, p in enumerate(params):
            grad_iterate[i] = grad_estimates[i] + hvp_delta[i] + alpha / 2 * norm_delta * delta[i]
        norm_grad_iterate = _compute_norm(grad_iterate)

        if norm_grad_iterate > eps / 2:
            break

    return delta


def _estimate_grad(params, l1, differentiable=True):
    grad_estimates = torch.autograd.grad(l1.mean(), params, create_graph=differentiable)
    return grad_estimates


def _estimate_hvp(params, l1, l2, v, grad_estimates=None):
    # Detach v from grad graph
    v = tuple(v_.detach() for v_ in v)

    if grad_estimates is None:
        grad_estimates = _estimate_grad(params, l1, differentiable=True)

    w = []
    for _ in l2:
        t = torch.autograd.grad(_, params, retain_graph=True)
        w.append(sum((t_ * v_).sum() for t_, v_ in zip(t, v)))

    Hvp1_torch = torch.autograd.grad(l1, params, grad_outputs=torch.tensor(w).to("cuda") / len(w), retain_graph=True)

    val = 0.
    for g_, v_ in zip(grad_estimates, v):
        val += torch.dot(g_.ravel(), v_.ravel())

    Hvp2_torch = torch.autograd.grad(val, params, retain_graph=True)
    Hvp_torch = tuple(h1 + h2 for h1, h2 in zip(Hvp1_torch, Hvp2_torch))
    return Hvp_torch


def _compute_dot_product(a, b):
    """
    :param a: List (Tensors)
    :param b: List (Tensors)
    :return: dot product
    """
    assert len(a) == len(b)

    res = sum((a_.detach() * b_.detach()).sum() for a_, b_ in zip(a, b))
    return res.item()


def _compute_norm(a):
    """
    :param a: List (Tensors)
    :return: dot product
    """
    res = _compute_dot_product(a, a)
    return torch.tensor(res).sqrt().item()
