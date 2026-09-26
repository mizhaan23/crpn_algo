import flax.linen as nn
import jax.numpy as jnp
from distrax import Normal
from typing import Sequence


def get_activation(act: str):
    if act == "tanh":
        return nn.tanh
    elif act == "relu":
        return nn.relu
    elif act == "softmax":
        return lambda x: nn.softmax(x, axis=-1)
    raise ValueError(f"Unknown activation: {act}")


class MLP(nn.Module):
    output_dim: int
    hidden_sizes: Sequence[int] = (32, 32)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = act_fn(x)
        return nn.Dense(self.output_dim)(x)


class GaussianMLPPolicy(nn.Module):
    output_dim: int
    hidden_sizes: Sequence[int] = (32, 32)
    activation: str = "tanh"
    action_scale: float = 1.0
    min_std: float = 1e-6

    @nn.compact
    def __call__(self, x):
        mean = MLP(output_dim=self.output_dim, hidden_sizes=self.hidden_sizes, activation=self.activation)(x)
        logstd = MLP(output_dim=self.output_dim, hidden_sizes=self.hidden_sizes, activation=self.activation)(x)
        return mean, logstd

    def get_action(self, params, x, key):
        mean, logstd = self.apply(params, x)
        action_mean = self.action_scale * jnp.tanh(mean)
        action_logstd = jnp.clip(jnp.asarray(logstd), -10.0, 2.0)
        action_std = jnp.maximum(jnp.exp(action_logstd), self.min_std)
        dist = Normal(loc=action_mean, scale=action_std)
        action = dist.sample(seed=key)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def get_log_prob(self, params, x, action):
        mean, logstd = self.apply(params, x)
        action_mean = self.action_scale * jnp.tanh(mean)
        action_logstd = jnp.clip(jnp.asarray(logstd), -10.0, 2.0)
        action_std = jnp.maximum(jnp.exp(action_logstd), self.min_std)
        dist = Normal(loc=action_mean, scale=action_std)
        return dist.log_prob(action).sum(-1)

