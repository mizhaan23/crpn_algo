import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from agent import LinearRLAgent


class LinearPolySGD(LinearRLAgent):
    def __init__(self, envs, lr=0.01, normalize_returns=False, poly_degree=1, set_bias=False):
        super().__init__(envs)

        self.lr = lr
        self.featurize = PolynomialFeatures(degree=poly_degree, include_bias=set_bias, order='F')
        self.featurize.fit(self.observation_space.sample().reshape(1, -1))

        self.num_features = self.featurize.n_output_features_
        self.num_params = self.num_features * self.action_space.n

        np.random.seed(1234)
        self.params = np.random.randn(self.num_params, ).astype(np.float32)
        self.normalize_returns = normalize_returns

    def get_state(self, observation):
        return self.featurize.transform(observation)  # (bs, state_dim)

    def get_action(self, state):
        P = self.params.reshape((self.num_features, self.action_space.n), order="F")  # (state_dim, num_actions)
        logits = state @ P  # (bs, state_dim) x (state_dim, num_actions)
        action_prob = softmax(logits, axis=1)  # (bs, num_actions)

        cdf = np.cumsum(action_prob, axis=1)  # (bs, num_actions)
        rvs = np.random.rand(action_prob.shape[0], )  # (bs, )

        # Sample action
        action = np.argmax(rvs[:, np.newaxis] < cdf, axis=1)
        return action, action_prob

    def learn(self, traj_info, dones, lr=None):
        GAMMA = self.discount_factor
        LEARNING_RATE = self.lr if lr is None else lr

        S = traj_info['states']
        A = traj_info['actions']
        P = traj_info['action_probs']
        R = traj_info['rewards']

        Y = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=self.normalize_returns)

        # Y = -Y for minimization of costs
        g, _ = self.compute_gradient_and_hessian_estimate(S, A, P, -Y, dones, compute_hessian=False)

        # make update
        self.params[:] = self.params - LEARNING_RATE * g
        return g, None, None
