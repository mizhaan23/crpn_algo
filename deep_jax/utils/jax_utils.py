"""
JAX port of deep/utils/_utils.py

This is the JAX equivalent of the PyTorch utility functions.
Key differences from PyTorch:
  - No device management (JAX handles it automatically)
  - No in-place mutation (all operations return new arrays)
  - jax.lax.scan replaces Python for-loops for JIT compatibility
  - PRNG keys must be passed explicitly (no global RNG state)
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom


# =============================================================================
# discount_cumsum — JAX port
# =============================================================================
#
# Original PyTorch code (deep/utils/_utils.py:61-75):
#
#   @torch.jit.script
#   def discount_cumsum(rewards, dones, gamma, normalize=True, device='cpu'):
#       discounted_rewards = torch.zeros_like(rewards).to(device)
#       cumulative_reward = torch.zeros_like(rewards[0]).to(device)
#       t = -1
#       for r in reversed(rewards):                          # <-- Python loop
#           cumulative_reward = r + cumulative_reward * gamma
#           discounted_rewards[t, :] = cumulative_reward     # <-- in-place mutation
#           t -= 1
#       ...
#
# JAX port below uses jax.lax.scan to replace the loop + mutation:

@jax.jit
def discount_cumsum(rewards, dones, gamma, normalize=True):
    """
    Compute discounted cumulative rewards (returns), scanning backwards in time.

    Args:
        rewards:   (T, N) array — reward at each timestep for N trajectories
        dones:     (T, N) bool array — whether episode was already done
        gamma:     float — discount factor (e.g. 0.99)
        normalize: bool — whether to standardize returns per trajectory
                   NOTE: This uses jax.lax.cond for JIT-compatibility.
                   Python `if normalize:` would fail inside @jax.jit because
                   JAX traces the function — it doesn't know the value of
                   `normalize` at trace time. jax.lax.cond defers the branch
                   decision to runtime.

    Returns:
        (T, N) array of discounted returns, masked by ~dones
    """
    # ---- Step 1: Reverse discounted cumsum via jax.lax.scan ----
    #
    # jax.lax.scan(f, init_carry, xs) applies f sequentially:
    #   carry_1, y_0 = f(init_carry, xs[0])
    #   carry_2, y_1 = f(carry_1, xs[1])
    #   ...
    # Returns (final_carry, stacked_ys)
    #
    # We flip rewards so scan processes time BACKWARDS.

    def scan_body(cumulative_reward, reward_t):
        # cumulative_reward: (N,) — running total
        # reward_t:          (N,) — reward at this timestep
        new_cumulative = reward_t + cumulative_reward * gamma
        return new_cumulative, new_cumulative  # (carry, output)

    _, discounted_rewards = jax.lax.scan(
        scan_body,
        init=jnp.zeros_like(rewards[0]),    # initial carry: (N,) zeros
        xs=jnp.flip(rewards, axis=0),       # reverse time axis
    )

    # Flip back to original time order
    discounted_rewards = jnp.flip(discounted_rewards, axis=0)

    # ---- Step 2: Normalize per trajectory (optional) ----
    #
    # We use jax.lax.cond instead of Python `if normalize:` because
    # inside @jax.jit, Python if-statements are evaluated at TRACE TIME.
    # jax.lax.cond evaluates the condition at RUNTIME.

    def _normalize(dr):
        """Standardize returns per trajectory using only pre-done timesteps."""
        # Find where each trajectory first becomes done
        first_done = jnp.argmax(dones.astype(jnp.int32), axis=0)  # (N,)

        # PyTorch slicing logic: m = first_done - 1, then mask for t < m
        m = first_done
        time_indices = jnp.arange(rewards.shape[0])[:, None]  # (T, 1)
        before_done_mask = time_indices < m[None, :]  # (T, N)

        # Compute mean and std only over valid (before-done) portion
        masked_returns = jnp.where(before_done_mask, dr, 0.0)
        counts = jnp.maximum(before_done_mask.sum(axis=0, keepdims=True), 1)

        mean = masked_returns.sum(axis=0, keepdims=True) / counts
        
        # PyTorch uses Bessel's correction (N-1) for std
        counts_bessel = jnp.maximum(counts - 1, 1)
        var = (jnp.where(before_done_mask, (dr - mean) ** 2, 0.0)
               .sum(axis=0, keepdims=True) / counts_bessel)
        std = jnp.sqrt(var) + 1e-9

        return (dr - mean) / std

    def _identity(dr):
        return dr

    discounted_rewards = jax.lax.cond(
        normalize, _normalize, _identity, discounted_rewards
    )

    # ---- Step 3: Mask out post-done timesteps ----
    return discounted_rewards * ~dones
