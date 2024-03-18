import time
import argparse
import numpy as np
import gymnasium as gym
import joblib
import random
import os
import inspect
from datetime import datetime
from distutils.util import strtobool
import shutil

from algo import LinearPolyCRPN
import wandb


def simulate_trajectories(envs, agent, policy, horizon):
    n = envs.num_envs
    observation_dim = envs.single_observation_space.shape
    state_dim = (agent.num_features,)
    action_dim = envs.single_action_space.shape
    num_actions = (envs.single_action_space.n,)  # number of discrete actions

    # Initializing simulation matrices for the given batched episode
    observations = np.zeros((horizon, n) + observation_dim, dtype=np.float32)
    states = np.zeros((horizon, n) + state_dim, dtype=np.float32)
    actions = np.zeros((horizon, n) + action_dim, dtype=np.int32)
    action_probs = np.zeros((horizon, n) + num_actions, dtype=np.float32)
    rewards = np.zeros((horizon, n), dtype=np.float32)
    dones = np.ones((horizon, n), dtype=bool)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    m = None
    for t in range(horizon):

        state = agent.get_state(obs)  # (bs, state_dim)
        action, action_prob = policy(obs)  # (bs, ), (bs, action_dim)

        observations[t] = obs
        states[t] = state
        actions[t] = action
        action_probs[t] = action_prob
        dones[t] = done

        obs, reward, terminated, truncated, info = envs.step(action)
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`

        reward = reward * ~done
        rewards[t] = reward

        if done.all():
            m = t
            break

    cum_discounted_rewards = agent.discount_cumsum(rewards, dones, gamma=0.99, normalize=False)
    cum_discounted_rewards = np.array(cum_discounted_rewards).astype(np.float32)
    mean_episode_return = np.sum(cum_discounted_rewards, axis=0) / np.sum(~dones, axis=0)

    traj_info = {
        'observations': observations[:m],
        'states': states[:m],
        'actions': actions[:m],
        'action_probs': action_probs[:m],
        'rewards': rewards[:m],
        'cum_discounted_rewards': cum_discounted_rewards[:m],
        'mean_episode_return': mean_episode_return,
    }

    return traj_info, dones[:m], np.sum(rewards, axis=0), mean_episode_return


def parse_args():
    parser = argparse.ArgumentParser()

    # environment specific args
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--env-seed", type=int, default=1,
                        help="the seed of the gym environment")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed of all rngs")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False,
                        help="the id of the gym environment")

    # agent specific args
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="the regularization parameter of the CRPN algorithm")
    parser.add_argument("--normalize-returns", type=lambda x: bool(strtobool(x)), default=True,
                        help="will normalize the returns by standard scaling")
    parser.add_argument("--poly-degree", type=int, default=1,
                        help="the max degree for polynomial features")
    parser.add_argument("--poly-bias", type=lambda x: bool(strtobool(x)), default=False,
                        help="whether to include bias for polynomial features")

    # simulation specific args
    parser.add_argument("--max-timesteps", type=int, default=500,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="total update epochs for the policy")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="the number of parallel game environments")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be saved locally")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be tracked on wandb")

    args = parser.parse_args()
    return args


def policy(observation, agent, parameters):
    agent.params[:] = parameters
    return agent.get_action(agent.get_state(observation))


def make_env(gym_id, idx, capture_video, run_name, args):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                b = args.num_updates
                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=lambda x: x % (2 * b // 10) == 0,
                )
        return env
    return thunk


def make_env1(gym_id, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, f"videos/{run_name}", name_prefix="rl-video-final",
        )
        return env

    return thunk


def get_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(args.alpha)}__{int(time.time())}"

    if args.track:
        config = args
        run = wandb.init(
            # Set the project where this run will be logged
            project="rl-crpn-team-research",
            entity='rlwork',
            # Track hyperparameters and run metadata
            monitor_gym=True,
            config=config,
            save_code=True,
        )

    # envs = gym.vector.make(args.gym_id, num_envs=args.batch_size, render_mode="rgb_array")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(args.gym_id) for i in range(args.batch_size)],
        copy=False,
    )

    if not args.env_seed == -1:
        envs.reset(seed=args.env_seed)
        envs.action_space.seed(seed=args.env_seed)
        envs.observation_space.seed(seed=args.env_seed)

    if args.capture_video:
        # Delete video_folder
        if os.path.exists(f'videos/{args.exp_name}'):
            shutil.rmtree(f'videos/{args.exp_name}')
            os.makedirs(f'videos/{args.exp_name}')

    agent = LinearPolyCRPN(
        envs, alpha=args.alpha,
        normalize_returns=args.normalize_returns,
        poly_degree=args.poly_degree, set_bias=args.poly_bias
    )

    # Simulation parameters
    max_timesteps = args.max_timesteps
    num_iterations = args.num_updates

    simulation_rewards = []
    simulation_returns = []
    delta_norm_squared, grad_norm_squared = [], []
    last_video = ''
    for i in range(num_iterations):

        t_ = time.time()
        params = agent.params.copy()
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs, agent, policy=lambda x: policy(x, agent, params), horizon=max_timesteps
        )
        t1 = time.time() - t_

        t_ = time.time()
        curr_grad, curr_Hess, opt_res = agent.learn(traj_info, dones)
        t2 = time.time() - t_

        simulation_rewards += list(episodic_rewards)
        simulation_returns += list(episodic_returns)
        grad_norm_squared.append(np.linalg.norm(curr_grad))
        delta_norm_squared.append(np.linalg.norm(opt_res.x))

        avg_traj_length = dones.shape[0]

        print(f"Iteration {i}, Reward: {np.mean(episodic_rewards)}, Delta norm squared: {np.linalg.norm(curr_grad) ** 2}, T1: {np.round(t1 / avg_traj_length * 1000, 3)}, "
              f"T2:{np.round(t2 / avg_traj_length * 1000, 3)}", end="\r")

        if args.track:
            if args.capture_video:
                curr_video = get_mp4_files(f"videos/{args.exp_name}")[-1]
                if curr_video != last_video:
                    v = wandb.Video(
                        data_or_path=curr_video,
                    )
                    wandb.log({'videos': v})
                last_video = curr_video

            # SOME ANALYSIS
            c1 = opt_res.x
            c2 = curr_grad

            sm = np.dot(c1, c2) / (np.linalg.norm(c1)*np.linalg.norm(c2))
            sr = np.linalg.norm(c1) / np.linalg.norm(c2)

            for rew, ret in zip(episodic_rewards, episodic_returns):
                wandb.log({'rewards': rew, 'returns': ret})
            wandb.log({'grad_norm_square': np.linalg.norm(curr_grad)**2, 'similarity': sm, 'step_ratio': sr, 'convergence': int(opt_res.success)})

    # Close environment
    envs.close()

    # FINAL RUN FOR VIDEO
    if args.capture_video:
        envs1 = gym.vector.SyncVectorEnv(
            [make_env1(args.gym_id, run_name=args.exp_name)]
        )
        params = agent.params.copy()
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs1, agent, policy=lambda x: policy(x, agent, params), horizon=max_timesteps
        )

        final_video = [i for i in get_mp4_files(f"videos/{args.exp_name}") if 'rl-video-final' in i][0]
        v = wandb.Video(
            data_or_path=final_video,
        )
        wandb.log({'final-video': v})

        # Close the final environment
        envs1.close()

    if args.save:

        curr_time = datetime.now().strftime('%Y%m%d%H%M%S')

        out_dict = {
            # simulation info
            "episodic_rewards": np.array(simulation_rewards, dtype=np.float32),
            "episodic_returns": np.array(simulation_returns, dtype=np.float32),
            "batch_size": int(args.batch_size),
            "horizon": int(args.max_timesteps),
            "nits": int(args.num_updates),
            "delta_norm_squared": np.array(delta_norm_squared, dtype=np.float32),
            "grad_norm_squared": np.array(grad_norm_squared, dtype=np.float32),

            # agent info
            "agent_name": agent.__class__.__name__,
            "trained_agent_params": agent.params,
            "agent_input_keys": list(dict(inspect.signature(agent.__class__).parameters).keys()),

            # env info
            "env_id": str(args.gym_id),
        }
        save_path = f"{os.path.basename(__file__).rstrip('.py')}/{'__'.join(run_name.split('__')[:-1])}"
        if not os.path.exists(f'./data/{save_path}'):
            os.makedirs(f'./data/{save_path}')

        for k, v in out_dict.items():
            joblib.dump(v, f"./data/{save_path}/{k}.data")
