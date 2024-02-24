import numpy as np


class LinearRLAgent:
    def __init__(self, envs, discount_factor=0.99):
        self.envs = envs
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.batch_size = envs.num_envs
        self.discount_factor = discount_factor

    def get_state(self, *args, **kwargs):
        raise NotImplementedError('Method `get_state` not defined!')

    def get_action(self, *args, **kwargs):
        raise NotImplementedError('Method `get_action` not defined!')

    def learn(self, *args, **kwargs):
        raise NotImplementedError('Method `learn` not defined!')

    def compute_gradient_and_hessian_estimate(self, S, A, P, Y, dones, compute_hessian=False):

        A_ = self.one_hot_encode_actions(A)
        Phi = self._mult1(A_, S)  # returns phi(s, a)
        EPhi = self._mult1(P, S)  # returns E[phi(s, .)] according to `P`

        nPhi = Phi - EPhi  # normalized phi
        grad_sum = self._mult1(1. * ~dones[..., np.newaxis], nPhi)
        grad_log_probs = self._mult1(Y[..., np.newaxis], nPhi)
        grad_est = np.sum(grad_log_probs, axis=(0, 1))
        grad_est = grad_est / self.batch_size

        hess_est = None
        if compute_hessian:
            # computing hessian estimate
            EPhiPhiT = self._mult2(self._diagonalize(P),
                                   S[..., np.newaxis] @ S[..., np.newaxis, :])  # returns E[Phi @ Phi.T]

            hess_est_1 = np.sum(grad_log_probs, axis=0)[..., np.newaxis] @ np.sum(grad_sum, axis=0)[..., np.newaxis, :]
            hess_est_1 = np.sum(hess_est_1, axis=0)

            hess_est_2 = self._mult2(Y[..., np.newaxis, np.newaxis],
                                     -EPhiPhiT + EPhi[..., np.newaxis] @ EPhi[..., np.newaxis, :])
            hess_est_2 = np.sum(hess_est_2, axis=(0, 1))
            hess_est = hess_est_1 + hess_est_2
            hess_est = hess_est / self.batch_size

        return grad_est, hess_est

    def one_hot_encode_actions(self, actions):
        a = actions.reshape(-1, order="F")
        b = np.zeros((a.size, self.action_space.n))
        b[np.arange(a.size), a] = 1
        B = b.reshape(actions.shape + (-1,), order="F")
        return B

    @staticmethod
    def discount_cumsum(rewards, dones, gamma, normalize=True):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = np.zeros_like(rewards[0])
        t = -1
        for r in rewards[::-1]:
            cumulative_reward = r + cumulative_reward * gamma  # Discount factor
            discounted_rewards[t, :] = cumulative_reward.copy()
            t -= 1
        if normalize:
            for i in range(rewards.shape[1]):
                m = np.argmax(dones[:, i]) - 1
                discounted_rewards[:, i] = (discounted_rewards[:, i] - discounted_rewards[:, i][:m].mean()) \
                                           / (discounted_rewards[:, i][:m].std() + 1e-9)
        return discounted_rewards * ~dones

    @staticmethod
    def _mult1(A, B):
        assert (A.shape[:-1] == B.shape[:-1]), f"{A.shape[:-1]}, {B.shape[:-1]}"
        out = np.einsum('...i,...j->...ij', A, B)
        out = out.reshape(A.shape[:-1] + (A.shape[-1] * B.shape[-1],), order="C")
        return out

    @staticmethod
    def _mult2(A, B):
        assert (A.shape[:-2] == A.shape[:-2]), f"{A.shape[:-2]}, {B.shape[:-2]}"
        out = np.einsum('...ij,...kl->...ikjl', A, B)
        out = out.reshape(A.shape[:-2] + (A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]), order="C")
        return out

    @staticmethod
    def _diagonalize(A):
        A_ = np.stack(tuple([A] * A.shape[-1]), axis=-1)
        return A_ * np.eye(A.shape[-1])
