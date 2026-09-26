import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple, NamedTuple
from .common import update_momentum


class ReinforceState(NamedTuple):
    m_grad: Any
    is_first: bool


def init(params: Any) -> ReinforceState:
    """Initialize REINFORCE momentum tracker."""
    m_init = jax.tree.map(jnp.zeros_like, params)
    return ReinforceState(m_grad=m_init, is_first=True)


def step(
    params: Any,
    opt_state: ReinforceState,
    l1_fn: Callable,
    l2_fn: Callable,
    num_envs: int,
    lr: float = 0.01,
    beta_1: float = 0.0,
    **kwargs
) -> Tuple[Any, ReinforceState]:
    """
    Standard first-order REINFORCE policy gradient update with optional gradient momentum (beta_1).
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad
        theta_{t+1} = theta_t - lr * m_t
    """
    mean_l1_fn = lambda p: l1_fn(p).mean()
    loss, grads = jax.value_and_grad(mean_l1_fn)(params)

    # Gradient momentum
    m_grad = update_momentum(opt_state.m_grad, grads, beta_1, opt_state.is_first)

    new_params = jax.tree.map(lambda p, g: p - lr * g, params, m_grad)
    new_opt_state = ReinforceState(m_grad=m_grad, is_first=False)

    return new_params, new_opt_state
