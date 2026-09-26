import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Any

from utils import simulate_trajectories, discount_cumsum
from optimizers import Optimizer, get_optimizer


def make_train_fn(env, env_params, agent, optimizer: Any, args):
    """
    Builds a single-seed JIT-compatible training function for any optimizer.
    
    Args:
        env: Gymnax environment
        env_params: Environment parameters
        agent: Policy agent network instance
        optimizer: Either an Optimizer instance or string name (e.g. 'acrpn', 'hapg')
        args: Parsed configuration namespace
        
    Returns:
        single_run(seed_key) -> (losses, mean_returns, Y_means, final_params)
    """
    if isinstance(optimizer, str):
        # Extract hyperparameters
        opt_kwargs = {}
        if hasattr(args, "alpha"):
            opt_kwargs["alpha"] = args.alpha
        if hasattr(args, "lr"):
            opt_kwargs["lr"] = args.lr
        if hasattr(args, "use_hessian"):
            opt_kwargs["use_hessian"] = args.use_hessian
        if hasattr(args, "beta_1"):
            opt_kwargs["beta_1"] = args.beta_1
        if hasattr(args, "beta_2"):
            opt_kwargs["beta_2"] = args.beta_2
        optimizer = get_optimizer(optimizer, **opt_kwargs)

    NUM_ENVS = args.batch_size
    MAX_TIMESTEPS = args.max_timesteps
    NUM_ITERATIONS = args.num_updates
    gamma = getattr(args, "discount_factor", 0.99)
    normalize_returns = getattr(args, "normalize_returns", True)
    use_std = getattr(args, "use_std", True)

    simulate_trajectories_fn = simulate_trajectories(env, env_params, policy=agent, horizon=MAX_TIMESTEPS)

    dummy_obs, _ = env.reset(jrandom.PRNGKey(0), env_params)
    obs_shape = dummy_obs.shape

    def single_run(seed_key):
        key_init, key_train = jrandom.split(seed_key, 2)
        params = agent.init(key_init, jnp.zeros(obs_shape))
        opt_state = optimizer.init(params)

        def step_fn(carry, _):
            params, opt_state, key = carry

            # 1. Rollout batch of parallel environments
            key, *env_keys = jrandom.split(key, NUM_ENVS + 1)
            env_keys = jnp.stack(env_keys)

            traj_info = simulate_trajectories_fn(env_keys, params)
            traj_info = jax.lax.stop_gradient(traj_info)

            obs = traj_info['obs']          # (horizon, NUM_ENVS, ...)
            actions = traj_info['actions']  # (horizon, NUM_ENVS)
            R_train = traj_info['rewards']        # (horizon, NUM_ENVS) - shaped reward for policy optimization
            R_eval = traj_info.get('raw_rewards', R_train) # (horizon, NUM_ENVS) - unshaped reward for evaluation / plotting
            dones = traj_info['dones']            # (horizon, NUM_ENVS)

            Y = discount_cumsum(R_train, dones, gamma=gamma, normalize=normalize_returns, use_std=use_std)
            mean_returns = R_eval.sum(axis=0).mean()
            Y_mean = Y.mean()

            # 2. Surrogate loss definitions evaluating only policy network
            def surrogate_losses(p):
                logP = agent.get_log_prob(p, obs, actions)
                mask = jnp.ones_like(dones) if MAX_TIMESTEPS == 1 else ~dones
                l1 = (logP * -Y).sum(0)
                l2 = (logP * mask).sum(0)
                return l1, l2

            l1_fn = lambda p: surrogate_losses(p)[0]
            l2_fn = lambda p: surrogate_losses(p)[1]

            # 3. Self-contained optimizer step
            new_params, new_opt_state = optimizer.step(params, opt_state, l1_fn, l2_fn, NUM_ENVS)
            reinforce_loss = l1_fn(params).mean()

            new_carry = (new_params, new_opt_state, key)
            return new_carry, (reinforce_loss, mean_returns, Y_mean)

        (final_params, _, _), (all_losses, all_mean_returns, all_Y_means) = jax.lax.scan(
            step_fn,
            init=(params, opt_state, key_train),
            xs=None,
            length=NUM_ITERATIONS,
        )
        return all_losses, all_mean_returns, all_Y_means, final_params

    return single_run
