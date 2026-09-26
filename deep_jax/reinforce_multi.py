import time
import json
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from reinforce import parse_args, make_env, make_agent, make_train_fn, run_multi_seeds


if __name__ == "__main__":
    args = parse_args(multi=True)
    start_timestamp = int(time.time())

    env, env_params = make_env(args.gym_id)
    agent = make_agent(env, env_params, args.hidden_sizes, getattr(args, "activation", "tanh"))
    single_run = make_train_fn(env, env_params, agent, args)

    master_key = jrandom.PRNGKey(args.seed)
    seed_keys = jrandom.split(master_key, args.num_seeds)

    total_transitions = args.num_seeds * args.batch_size * args.max_timesteps * args.num_updates
    print(f"Configuration: {args.num_seeds} seeds | {args.batch_size} envs/seed | {args.max_timesteps} horizon | {args.num_updates} updates | Mode: {args.mode.upper()}")
    print(f"Total environment steps: {total_transitions:,}")

    wall_start = time.perf_counter()
    all_losses, all_mean_returns, all_Y_means, (warmup_time, exec_time) = run_multi_seeds(
        single_run, seed_keys, mode=args.mode, chunk_size=getattr(args, "chunk_size", None)
    )
    wall_total = time.perf_counter() - wall_start

    # Print summary statistics
    final_returns = np.array(all_mean_returns[:, -1])
    peak_returns = np.array(jnp.max(all_mean_returns, axis=1))
    print(f"\n{'='*60}")
    print(f"REINFORCE | {args.gym_id} | {args.activation} | hidden={args.hidden_sizes}")
    print(f"Seeds: {args.num_seeds} | Final: {final_returns.mean():.2f} ± {final_returns.std():.2f}")
    print(f"Peak: {peak_returns.mean():.2f} ± {peak_returns.std():.2f}")
    print(f"Wall time: {wall_total:.1f}s (warmup={warmup_time:.1f}s, exec={exec_time:.1f}s)")
    print(f"{'='*60}")

    if args.save:
        import os
        os.makedirs("data", exist_ok=True)
        tag = f"reinforce_{args.gym_id}_{args.activation}_h{'_'.join(map(str,args.hidden_sizes))}_{args.num_seeds}seeds"
        np.savez(f"data/{tag}.npz",
                 returns=np.array(all_mean_returns),
                 losses=np.array(all_losses),
                 Y_means=np.array(all_Y_means))
        timing = {"tag": tag, "warmup_s": warmup_time, "exec_s": exec_time, "wall_s": wall_total,
                  "num_seeds": args.num_seeds, "final_mean": float(final_returns.mean()),
                  "final_std": float(final_returns.std()), "peak_mean": float(peak_returns.mean())}
        with open(f"data/{tag}_timing.json", "w") as f:
            json.dump(timing, f, indent=2)
        print(f"Saved: data/{tag}.npz and data/{tag}_timing.json")
