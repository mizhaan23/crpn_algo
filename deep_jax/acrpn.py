"""
Adaptive Cubic Regularized Policy Newton (ACRPN) — Standalone Runner & Module.
Refactored to use modular optimizers and engine.
"""

import jax
import jax.random as jrandom

from utils import (
    make_env,
    make_agent,
    parse_args,
    compile_and_run,
    run_multi_seeds,
    simulate_trajectories,
    discount_cumsum,
)
from optimizers import get_optimizer
import engine


def make_train_fn(env, env_params, agent, args):
    """Build single-seed ACRPN training function using modular engine."""
    alpha = getattr(args, "alpha", 1000.0)
    beta_1 = getattr(args, "beta_1", 0.0)
    beta_2 = getattr(args, "beta_2", 0.0)
    optimizer = get_optimizer("acrpn", alpha=alpha, beta_1=beta_1, beta_2=beta_2)
    return engine.make_train_fn(env, env_params, agent, optimizer, args)


def main():
    args = parse_args(default_exp_name="acrpn", multi=False)
    env, env_params = make_env(args.gym_id)
    if hasattr(env_params, "max_steps_in_episode"):
        env_params = env_params.replace(max_steps_in_episode=args.max_timesteps)

    agent = make_agent(env, env_params, args.hidden_sizes, args.activation, getattr(args, "policy_type", None))
    single_run = make_train_fn(env, env_params, agent, args)

    key = jrandom.PRNGKey(args.seed)
    results, warmup, exec_t = compile_and_run(single_run, key, label=f"ACRPN ({args.gym_id})")
    losses, returns, Y_means, _ = results
    print(f"\nFinal Average Return: {returns[-1]:.2f}")


if __name__ == "__main__":
    main()
