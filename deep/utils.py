import numpy as np
import torch


def compute_grad_log_prob(s, a, p):
    g = 0.
    for a_, p_ in zip(range(2), p.ravel()):
        g -= p_ * phi(s, a_)
    g += phi(s, a)
    return g


def compute_hess_log_prob(s, a, p):
    H = 0.
    for a_, p_ in zip(range(2), p.ravel()):
        H -= p_ * phi(s, a_)[:, np.newaxis] @ compute_grad_log_prob(s, a_, p)[np.newaxis, :]
    return H


def phi(s, a):
    '''
    Outputs phi(s, a) using polynomial features.
    '''
    out = np.zeros((s.size, 2), dtype=np.float32, order='F')
    out[:, a] = s.ravel()
    return out.reshape(-1, order='F')


# def discount_cumsum(rewards, dones, gamma, normalize=True):
#     discounted_rewards = np.zeros_like(rewards)
#     cumulative_reward = np.zeros_like(rewards[0])
#     t = -1
#     for r in rewards[::-1]:
#         cumulative_reward = r + cumulative_reward * gamma  # Discount factor
#         discounted_rewards[t, :] = cumulative_reward.copy()
#         t -= 1
#     if normalize:
#         for i in range(rewards.shape[1]):
#             m = np.argmax(dones[:, i]) - 1
#             discounted_rewards[:, i] = discounted_rewards[:, i] / (discounted_rewards[:, i][:m].std() + 1e-9)
#     return discounted_rewards

def discount_cumsum(rewards, dones, gamma, normalize=True, device='cpu'):
    discounted_rewards = torch.zeros_like(rewards).to(device)
    cumulative_reward = torch.zeros_like(rewards[0]).to(device)
    t = -1
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * gamma  # Discount factor
        discounted_rewards[t, :] = cumulative_reward
        t -= 1
    if normalize:
        for i in range(rewards.shape[1]):
            m = np.argmax(dones[:, i].cpu()) - 1
            discounted_rewards[:, i] = discounted_rewards[:, i] / (discounted_rewards[:, i][:m].std() + 1e-9)
    return discounted_rewards


def _mult1(A, B):
    assert (A.shape[:-1] == B.shape[:-1]), f"{A.shape[:-1]}, {B.shape[:-1]}"
    out = torch.einsum('...i,...j->...ij', A, B)
    out = out.reshape(A.shape[:-1] + (A.shape[-1] * B.shape[-1],), order="C")
    return out
