import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from scipy.optimize import minimize
from ._agent import LinearRLAgent


class LinearPolyCRPN(LinearRLAgent):
    def __init__(self, envs, alpha=1e4, normalize_returns=False, poly_degree=1, set_bias=False):
        super().__init__(envs)

        self.alpha = alpha
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
        p = self.params.reshape((self.num_features, self.action_space.n), order="F")  # (state_dim, num_actions)
        logits = state @ p  # (bs, state_dim) x (state_dim, num_actions)
        action_prob = softmax(logits, axis=1)  # (bs, num_actions)

        cdf = np.cumsum(action_prob, axis=1)  # (bs, num_actions)
        rvs = np.random.rand(action_prob.shape[0], )  # (bs, )

        # Sample action
        action = np.argmax(rvs[:, np.newaxis] < cdf, axis=1)
        return action, action_prob

    def learn(self, traj_info, dones, alp=None):
        GAMMA = self.discount_factor
        ALPHA = self.alpha if alp is None else alp

        S = traj_info['states']
        A = traj_info['actions']
        P = traj_info['action_probs']
        R = traj_info['rewards']

        Y = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=self.normalize_returns)

        # Y = -Y for minimization of costs
        g, H = self.compute_gradient_and_hessian_estimate(S, A, P, -Y, dones, compute_hessian=True)

        # Compute the optima for the cubic-regularized sub-problem
        v0 = np.random.randn(self.num_params,)
        result = minimize(
            self._fg, v0, method='Newton-CG', jac=True, hess=self._hess, args=(H, g, ALPHA),
            tol=np.finfo(np.float32).eps, options={'maxiter': 500}
        )

        # make update
        self.params[:] = self.params + result.x
        return g, H, result

    @staticmethod
    def _fg(v, H, g, alpha):
        n = np.linalg.norm(v)
        Hv = H @ v
        s = np.dot(g, v) + .5 * np.dot(Hv, v) + alpha / 6 * n ** 3
        j = g + Hv + alpha / 2 * n * v
        return s, j

    @staticmethod
    def _hess(v, H, g, alpha):
        n = np.linalg.norm(v)
        return H + alpha / 2 * (v @ v.T / n + n * np.eye(len(v)))
