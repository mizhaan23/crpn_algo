import numpy as np
import gymnax
from gymnax.environments import spaces

from policies.categorical_mlp_policy import CategoricalMLPPolicy
from policies.categorical_linear_policy import CategoricalLinearPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy
from policies.minatar_conv_policy import MinAtarConvPolicy


def make_env(gym_id: str):
    """Create Gymnax environment and default parameters."""
    if gym_id in ["CurvedCorridor-v0", "CurvedCorridor"]:
        from envs.curved_corridor import CurvedCorridorEnv
        env = CurvedCorridorEnv()
        return env, env.default_params
    elif gym_id in ["CliffWalking-v0", "CliffWalking"]:
        from envs.cliffwalking import CliffWalkingEnv
        env = CliffWalkingEnv(height=4, width=12)
        return env, env.default_params
    elif gym_id in ["CliffWalking4x6-v0", "CliffWalking4x6"]:
        from envs.cliffwalking import CliffWalkingEnv
        env = CliffWalkingEnv(height=4, width=6)
        return env, env.default_params
    elif gym_id in ["CliffWalking4x4-v0", "CliffWalking4x4"]:
        from envs.cliffwalking import CliffWalkingEnv
        env = CliffWalkingEnv(height=4, width=4)
        return env, env.default_params
    elif gym_id in ["RosenbrockBandit-v0", "RosenbrockBandit", "Rosenbrock"]:
        from envs.contextual_bandits import RosenbrockBanditEnv
        env = RosenbrockBanditEnv()
        return env, env.default_params
    elif gym_id in ["SaddlePointBandit-v0", "SaddlePointBandit", "SaddlePoint"]:
        from envs.contextual_bandits import SaddlePointBanditEnv
        env = SaddlePointBanditEnv()
        return env, env.default_params
    return gymnax.make(gym_id)


def make_agent(env, env_params, hidden_sizes=(), activation="relu", policy_type=None):
    """Instantiate appropriate policy agent according to environment action and observation spaces."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)

    # 1. MinAtar 3D spatial grids
    if len(obs_space.shape) == 3:
        if policy_type == "mlp" or (policy_type is None and len(tuple(hidden_sizes)) > 0):
            print(f"MinAtar Flattened MLP Action Space (obs shape: {obs_space.shape}, hidden: {hidden_sizes}, act: {activation})")
            return CategoricalMLPPolicy(output_dim=action_space.n, hidden_sizes=tuple(hidden_sizes), activation=activation)
        else:
            print(f"MinAtar ConvNet Action Space (obs shape: {obs_space.shape})")
            return MinAtarConvPolicy(output_dim=action_space.n)

    # 2. Discrete action spaces (CartPole, Acrobot, MountainCar)
    if isinstance(action_space, spaces.Discrete):
        print(f"Discrete Action Space (obs shape: {obs_space.shape}, act: {activation})")
        if len(tuple(hidden_sizes)) == 0:
            return CategoricalLinearPolicy(output_dim=action_space.n)
        return CategoricalMLPPolicy(output_dim=action_space.n, hidden_sizes=tuple(hidden_sizes), activation=activation)

    # 3. Continuous action spaces (Pendulum, MountainCarContinuous) -> Always enforce tanh for Gaussian policies
    elif isinstance(action_space, spaces.Box):
        action_scale = float(np.max(action_space.high)) if hasattr(action_space, "high") else 1.0
        print(f"Continuous Action Space (obs shape: {obs_space.shape}, scale: {action_scale}) -> Enforcing 'tanh' activation for Gaussian policy")
        return GaussianMLPPolicy(output_dim=int(np.prod(action_space.shape)), hidden_sizes=tuple(hidden_sizes), activation="tanh", action_scale=action_scale)

    raise NotImplementedError(f"Unsupported action space: {action_space}")
