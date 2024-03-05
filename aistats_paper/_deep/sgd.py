import argparse
import math
import os
import random
import shutil
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import utils


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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True,
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
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True,
                        help="if toggled, this experiment will be saved locally")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True,
                        help="if toggled, this experiment will be saved locally")

    # cuda stuff
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    args = parser.parse_args()
    return args


def simulate_trajectories(envs, agent, horizon, device):
    n = envs.num_envs
    observation_dim = envs.single_observation_space.shape
    action_dim = envs.single_action_space.shape
    num_actions = (envs.single_action_space.n,)  # number of discrete actions

    # Initializing simulation matrices for the given batched episode
    observations = torch.zeros((horizon, n) + observation_dim, dtype=torch.float32).to(device)
    actions = torch.zeros((horizon, n) + action_dim, dtype=torch.int32).to(device)
    action_probs = torch.zeros((horizon, n) + num_actions, dtype=torch.float32).to(device)

    log_probs = torch.zeros((horizon, n) + action_dim, dtype=torch.float32).to(device)
    rewards = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    dones = torch.ones((horizon, n), dtype=bool).to(device)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    m = None
    for t in range(horizon):

        obs = torch.tensor(obs).to(device)
        action_prob = agent(obs)  # (bs, ), (bs, action_dim)
        dist = Categorical(action_prob)
        action = dist.sample()

        # print(Categorical(action_prob).sample())
        observations[t] = obs
        actions[t] = action
        action_probs[t] = action_prob

        log_probs[t] = dist.log_prob(action)
        dones[t] = torch.tensor(done).to(device)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().detach().numpy())
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        reward = reward * ~done
        rewards[t] = torch.tensor(reward).to(device)

        if done.all():
            m = t
            break

    cum_discounted_rewards = utils.discount_cumsum(rewards, dones, gamma=0.99, normalize=False, device=device)
    mean_episode_return = torch.sum(cum_discounted_rewards, axis=0) / torch.sum(~dones, axis=0)

    traj_info = {
        'observations': observations[:m],
        'actions': actions[:m],
        'action_probs': action_probs[:m],
        'log_probs': log_probs[:m],
        'rewards': rewards[:m],
        'cum_discounted_rewards': cum_discounted_rewards[:m],
        'mean_episode_return': mean_episode_return,
    }

    return traj_info, dones[:m], torch.sum(rewards, axis=0), mean_episode_return


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    np.random.seed(1234)
    params = np.random.randn(layer.weight.data.numel(), ).astype(np.float32)
    layer.weight.data = torch.from_numpy(params.reshape(layer.weight.data.shape, order="C"))
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        torch.manual_seed(0)
        self.fc1 = layer_init(
            nn.Linear(
                np.array(envs.single_observation_space.shape).prod(),
                envs.single_action_space.n,
                bias=False
            )
        )
        # self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return F.softmax(x, dim=-1)


if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            # Set the project where this run will be logged
            project="rl-crpn-team-research",
            entity="rlwork",
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, i, args.capture_video, run_name, args=args) for i in range(args.batch_size)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if not args.env_seed == -1:
        envs.reset(seed=args.env_seed)
        envs.action_space.seed(seed=args.env_seed)
        envs.observation_space.seed(seed=args.env_seed)

    if args.capture_video:
        # Delete video_folder
        if os.path.exists(f'videos/{args.exp_name}'):
            shutil.rmtree(f'videos/{args.exp_name}')
            os.makedirs(f'videos/{args.exp_name}')

    agent = Agent(envs).to(device)
    optimizer = optim.SGD(agent.parameters(), lr=math.sqrt(1 / args.alpha))

    # Simulation parameters
    max_timesteps = args.max_timesteps
    num_iterations = args.num_updates

    simulation_rewards = []
    simulation_returns = []
    last_video = ''
    for i in range(num_iterations):

        t_ = time.time()
        traj_info, dones, episodic_rewards, episodic_returns = simulate_trajectories(
            envs, agent=agent, horizon=500, device=device
        )
        t1 = time.time() - t_

        # Agent learns here
        t_ = time.time()

        logP, R = traj_info['log_probs'], traj_info['rewards']
        Y = utils.discount_cumsum(R, dones, gamma=0.99, normalize=args.normalize_returns, device=device)
        print(logP.shape, Y.shape)

        l1 = (logP * -Y).sum(0)
        l2 = (logP * ~dones).sum(0)

        optimizer.zero_grad()
        loss = l1.mean()
        loss.backward()
        optimizer.step()

        t2 = time.time() - t_

        simulation_rewards += list(episodic_rewards)
        simulation_returns += list(episodic_returns)

        avg_traj_length = dones.shape[0]

        print(f"Iteration {i}, Reward: {episodic_rewards.mean()}, T1: {np.round(t1 / avg_traj_length * 1000, 3)}, "
              f"T2:{np.round(t2 / avg_traj_length * 1000, 3)}", end="\r")

        if args.track:

            for t, r in enumerate(zip(episodic_rewards, episodic_returns)):
                wandb.log({'rewards': r[0], 'returns': r[1]})
                writer.add_scalar("charts/rewards", r[0].item(), i+t)
                writer.add_scalar("charts/returns", r[1].item(), i+t)
                writer.add_scalar("losses/log_loss", l1[t].item(), i+t)

    # Close environment
    envs.close()
    writer.close()

    # if args.save:
    #
    #     curr_time = datetime.now().strftime('%Y%m%d%H%M%S')
    #
    #     out_dict = {
    #         # simulation info
    #         "episodic_rewards": simulation_rewards,
    #         "episodic_returns": simulation_returns,
    #         "batch_size": args.batch_size,
    #         "horizon": args.max_timesteps,
    #         "nits": args.num_updates,
    #
    #         # agent info
    #         "agent_name": agent.__class__.__name__,
    #         "trained_agent_params": agent.params,
    #         "agent_input_keys": list(dict(inspect.signature(agent.__class__).parameters).keys()),
    #
    #         # env info
    #         "env_id": args.gym_id,
    #     }
    #     save_path = f"{os.path.basename(__file__).rstrip('.py')}/{args.exp_name}"
    #     if not os.path.exists(f'./data/{save_path}'):
    #         os.makedirs(f'./data/{save_path}')
    #
    #     for k, v in out_dict.items():
    #         joblib.dump(v, f"./data/{save_path}/{args.exp_name}.{k}")
