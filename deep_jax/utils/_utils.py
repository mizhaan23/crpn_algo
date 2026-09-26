import jax
import jax.numpy as jnp
import jax.random as jrandom


def simulate_trajectories(env, env_params, policy, horizon):
    """
    Description: Returns a JIT'ed function to simulate trajectories.

    Args:
        env: The environment to simulate trajectories in.
        env_params: The parameters of the environment.
        policy: The policy to use for simulating trajectories.
        horizon: The number of steps to simulate trajectories for.

    Returns:
        A JIT'ed function that can be used to simulate trajectories.

    Example use case:
    simulate_trajectories_fn = simulate_trajectories(env, env_params, policy, horizon)
    traj_info = simulate_trajectories_fn(env_keys, params)
    """
    def _simulate_single(key, params):
        key, reset_key = jrandom.split(key, 2)
        obs, env_state = env.reset(key, env_params)
        done = jnp.array(False)

        def step_fn(carry, x):
            key, obs, env_state, done = carry

            key, step_key, action_key = jrandom.split(key, 3)
            action, log_prob = policy.get_action(params, obs, action_key)
            next_obs, next_env_state, reward, next_done, info = env.step(step_key, env_state, action)

            # Mask subsequent transitions after episode termination
            mask = ~done
            done = next_done | done

            raw_reward = info.get("raw_reward", reward) if isinstance(info, dict) else reward

            new_carry = key, next_obs, next_env_state, done
            return new_carry, (obs, action, log_prob, reward * mask, done, raw_reward * mask)

        final_carry, (obs, action, log_prob, reward, done, raw_reward) = jax.lax.scan(
            step_fn,
            init=(key, obs, env_state, done),
            xs=None,
            length=horizon,
        )
        return {'obs': obs, 'actions': action, 'log_probs': log_prob, 'rewards': reward, 'dones': done, 'raw_rewards': raw_reward}

    return jax.jit(jax.vmap(_simulate_single, in_axes=(0, None), out_axes=1))



@jax.jit
def discount_cumsum(rewards, dones, gamma: float, normalize: bool = False):
    """
    Description: Returns a JIT'ed function to calculate discounted cumulative sums.

    Args:
        rewards: The rewards to calculate discounted cumulative sums for.
        dones: The dones to calculate discounted cumulative sums for.
        gamma: The discount factor.
        normalize: Whether to normalize the discounted cumulative sums.

    Returns:
        A JIT'ed function that can be used to calculate discounted cumulative sums.

    Example use case:
    discount_cumsum = discount_cumsum_fn(rewards, dones, gamma, normalize)
    traj_info = discount_cumsum(rewards, dones, gamma, normalize)
    """
@jax.jit
def discount_cumsum(rewards, dones, gamma: float, normalize: bool = False, use_std: bool = True):
    """
    Description: Returns a JIT'ed function to calculate discounted cumulative sums.
    When normalize=True and use_std=False, subtracts mean baseline without dividing by std.
    For 1-step contextual bandits (rewards.shape[0] == 1), normalizes across the batch dimension.
    """
    if rewards.shape[0] == 1:
        mean = jnp.mean(rewards, axis=1, keepdims=True)
        ret_mean = rewards - mean
        std = jnp.std(rewards, axis=1, keepdims=True)
        denom = jax.lax.select(use_std, std + 1e-8, jnp.ones_like(std))
        norm_r = ret_mean / denom
        return jax.lax.cond(normalize, lambda: norm_r, lambda: rewards)

    def _single(reward, done, gamma, normalize=False, use_std=True):
        reversed_reward = reward[::-1]

        def step_fn(carry, r):
            new_carry = r + gamma*carry
            return new_carry, new_carry
        
        _, reversed_return = jax.lax.scan(step_fn, init=0., xs=reversed_reward)
        ret = reversed_return[::-1] * ~done

        def _normalize(ret):
            n = (~done).sum()
            safe_n = jnp.maximum(n, 1)
            mean = jnp.dot(ret, ~done) / safe_n
            ret_mean = (ret - mean) * ~done
            var = (ret_mean**2).sum() / jnp.maximum(n - 1, 1)  # bessel's correction
            std = jnp.sqrt(jnp.maximum(var, 0.0))
            denom = jax.lax.select(use_std, std + 1e-8, 1.0)
            return ret_mean / denom
        
        return jax.lax.cond(normalize, _normalize, lambda x: x, ret) * ~done
    
    return jax.vmap(_single, in_axes=(1, 1, None, None, None), out_axes=1)(rewards, dones, gamma, normalize, use_std)