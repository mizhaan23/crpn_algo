import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple, NamedTuple
from .common import pytree_dot, pytree_norm, compute_policy_hvp, update_momentum


class ACRPNState(NamedTuple):
    m_grad: Any
    u_hvp: Any
    is_first: bool


def init(params: Any) -> ACRPNState:
    """Initialize ACRPN momentum trackers for gradients (beta_1) and Hessians (beta_2)."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return ACRPNState(m_grad=zeros, u_hvp=zeros, is_first=True)


def step(
    params: Any,
    opt_state: ACRPNState,
    l1_fn: Callable,
    l2_fn: Callable,
    num_envs: int,
    alpha: float = 1000.0,
    beta_1: float = 0.0,
    beta_2: float = 0.0,
    use_hessian: bool = True,
    **kwargs
) -> Tuple[Any, ACRPNState]:
    """
    Adaptive Cubic Regularized Policy Newton (ACRPN) Cauchy step with gradient & Hessian momentum.
    When use_hessian=False, reduces directly to the first-order Baseline Cauchy step.
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad_t
        u_t = beta_2 * u_{t-1} + (1 - beta_2) * (H_t m_t)
        beta_curv = <m_t, u_t> / (alpha * ||m_t||^2) if use_hessian else 0
        R_c = -beta_curv + sqrt(beta_curv^2 + 2 * ||m_t|| / alpha)
        theta_{t+1} = theta_t - R_c * (m_t / ||m_t||)
    """
    mean_l1_fn = lambda p: l1_fn(p).mean()
    grad = jax.grad(mean_l1_fn)(params)

    # 1. Gradient momentum (beta_1)
    m_t = update_momentum(opt_state.m_grad, grad, beta_1, opt_state.is_first)
    grad_norm = pytree_norm(m_t)

    # 2. Hessian curvature momentum (beta_2)
    if use_hessian:
        hg = compute_policy_hvp(l1_fn, l2_fn, params, m_t, num_envs)
        u_t = update_momentum(opt_state.u_hvp, hg, beta_2, opt_state.is_first)
        beta_curv = pytree_dot(m_t, u_t) / (alpha * grad_norm * grad_norm + 1e-8)
    else:
        u_t = opt_state.u_hvp
        beta_curv = 0.0

    R_c = -beta_curv + jnp.sqrt(beta_curv * beta_curv + 2.0 * grad_norm / alpha)
    new_params = jax.tree.map(
        lambda p, g: p - R_c * g / (grad_norm + 1e-8),
        params,
        m_t
    )
    new_opt_state = ACRPNState(m_grad=m_t, u_hvp=u_t, is_first=False)
    return new_params, new_opt_state
