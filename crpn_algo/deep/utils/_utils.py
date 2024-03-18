import time
import numpy as np
import torch


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to execute {func.__name__}: {elapsed_time} seconds")
        return result

    return wrapper


def simulate_trajectories(envs, policy, horizon, device):
    n = envs.num_envs

    # Initializing simulation matrices for the given batched episode
    log_probs = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    rewards = torch.zeros((horizon, n), dtype=torch.float32).to(device)
    dones = torch.ones((horizon, n), dtype=bool).to(device)

    obs, _ = envs.reset()
    done = np.zeros((n,), dtype=bool)  # e.g. [False, False, False]
    T = None

    for t in range(horizon):
        obs = torch.tensor(np.float32(obs)).to(device)

        action, log_prob = policy.get_action(obs)

        log_probs[t] = log_prob
        dones[t] = torch.tensor(done).to(device)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().detach().numpy())
        done = done | (np.array(terminated) | np.array(truncated))

        # Modify rewards to NOT consider data points after `done`
        reward = reward * ~done
        rewards[t] = torch.tensor(reward).to(device)

        if done.all():
            T = t
            break

    cum_discounted_rewards = discount_cumsum(rewards, dones, gamma=0.99, normalize=False, device=device)
    mean_episode_return = torch.sum(cum_discounted_rewards, axis=0) / torch.sum(~dones, axis=0)

    traj_info = {
        'log_probs': log_probs[:T],
        'rewards': rewards[:T],
        'dones': dones[:T],
    }

    return traj_info, torch.sum(rewards, axis=0), mean_episode_return


@torch.jit.script
def discount_cumsum(rewards, dones, gamma: float, normalize: bool = True, device: torch.device = 'cpu') -> torch.Tensor:
    discounted_rewards = torch.zeros_like(rewards).to(device)
    cumulative_reward = torch.zeros_like(rewards[0]).to(device)
    t = -1
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * gamma  # Discount factor
        discounted_rewards[t, :] = cumulative_reward
        t -= 1
    if normalize:
        for i in range(rewards.shape[1]):
            m = torch.argmax(1. * dones[:, i]) - 1
            discounted_rewards[:, i] = (discounted_rewards[:, i] - discounted_rewards[:, i][:m].mean()) / (
                        discounted_rewards[:, i][:m].std() + 1e-9)
    return discounted_rewards * ~dones
