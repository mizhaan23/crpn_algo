import jax
import flax.linen as nn
from distrax import Categorical
from typing import Sequence


class CategoricalMLPPolicy(nn.Module):
    output_dim: int
    hidden_sizes: Sequence[int] = (32, 32)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        # Flatten spatial grid observations (e.g. MinAtar 10x10xC)
        if x.ndim >= 4:
            x = x.reshape(x.shape[:-3] + (-1,))
        elif x.ndim == 3 and x.shape[0] == 10 and x.shape[1] == 10:
            x = x.reshape((-1,))

        act_fn = nn.tanh if self.activation == "tanh" else (nn.relu if self.activation == "relu" else lambda z: nn.softmax(z, axis=-1))
        for size in self.hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = act_fn(x)
        return nn.Dense(features=self.output_dim)(x)

    def get_action(self, params, x, key):
        dist = Categorical(logits=self.apply(params, x))
        action = dist.sample(seed=key)
        return action, dist.log_prob(action)

    def get_log_prob(self, params, x, action):
        apply_fn = jax.checkpoint(self.apply)
        dist = Categorical(logits=apply_fn(params, x))
        return dist.log_prob(action)


