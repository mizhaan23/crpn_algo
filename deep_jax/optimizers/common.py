import jax
import jax.numpy as jnp
from typing import Callable, Any


def pytree_dot(a: Any, b: Any) -> jax.Array:
    """Computes the inner product <a, b> across all leaves of two matching pytrees."""
    leaves_prod = jax.tree.map(lambda x, y: jnp.sum(x * y), a, b)
    return sum(jax.tree.leaves(leaves_prod))


def pytree_norm(a: Any) -> jax.Array:
    """Computes the Euclidean (L2) norm across all leaves of a pytree."""
    leaves = jax.tree.leaves(a)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.vdot(x, x) for x in leaves])))


def pytree_scale(scalar: float, tree: Any) -> Any:
    """Multiplies every leaf in a pytree by a scalar."""
    return jax.tree.map(lambda x: scalar * x, tree)


def pytree_add(a: Any, b: Any) -> Any:
    """Element-wise addition of two matching pytrees."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def compute_policy_hvp(l1_fn: Callable, l2_fn: Callable, params: Any, v: Any, num_envs: int) -> Any:
    """
    Computes exact forward-over-reverse Hessian-Vector Product for policy gradient surrogate loss:
        H v = H_loss v + H_dist v
        hvp1 = (grad^2 mean(l1)) @ v
        hvp2 = (1 / B) * sum_i [ (grad l2_i @ v) * grad l1_i ]
    """
    mean_l1_fn = lambda p: l1_fn(p).mean()
    grad_l1_fn = jax.grad(mean_l1_fn)
    
    # 1. Loss curvature via forward-mode JVP of gradient
    _, hvp1 = jax.jvp(grad_l1_fn, (params,), (v,))
    
    # 2. Distribution shift cross-term via forward JVP of l2 + reverse VJP of l1
    _, jvp_l2 = jax.jvp(l2_fn, (params,), (v,))
    _, vjp_l1_fn = jax.vjp(l1_fn, params)
    hvp2 = vjp_l1_fn(jvp_l2 / num_envs)[0]
    
    return pytree_add(hvp1, hvp2)


def update_momentum(prev_m: Any, new_v: Any, beta: float, is_first: Any = False) -> Any:
    """
    Exponential Moving Average momentum update with unbiased initialization:
        m_t = beta * m_{t-1} + (1 - beta) * v_t
    If beta == 0.0, directly returns new_v (zero momentum overhead).
    If is_first is True, initializes directly to new_v.
    """
    if beta == 0.0:
        return new_v
    
    return jax.lax.cond(
        is_first,
        lambda: new_v,
        lambda: jax.tree.map(lambda m, v: beta * m + (1.0 - beta) * v, prev_m, new_v)
    )
