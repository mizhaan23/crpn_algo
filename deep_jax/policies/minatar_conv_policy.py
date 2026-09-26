import flax.linen as nn
import jax.numpy as jnp
from distrax import Categorical


class MinAtarConvPolicy(nn.Module):
    """
    Convolutional Policy for MinAtar environments (10x10xC binary observation grids).
    Architecture:
      Conv(16 filters, 3x3 kernel, valid padding) -> ReLU
      Flatten -> Dense(64) -> ReLU
      Dense(output_dim) -> Categorical Action Distribution
    """
    output_dim: int
    conv_features: int = 16
    dense_features: int = 64

    @nn.compact
    def __call__(self, x):
        is_unbatched = (x.ndim == 3)
        if is_unbatched:
            x = jnp.expand_dims(x, 0)
        orig_shape = x.shape
        if x.ndim > 4:
            x = x.reshape((-1,) + orig_shape[-3:])

        x = nn.Conv(features=self.conv_features, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.dense_features)(x)
        x = nn.relu(x)
        out = nn.Dense(features=self.output_dim)(x)

        if is_unbatched:
            return out[0]
        if len(orig_shape) > 4:
            return out.reshape(orig_shape[:-3] + (self.output_dim,))
        return out

    def get_action(self, params, x, key):
        dist = Categorical(logits=self.apply(params, x))
        action = dist.sample(seed=key)
        return action, dist.log_prob(action)

    def get_log_prob(self, params, x, action):
        dist = Categorical(logits=self.apply(params, x))
        return dist.log_prob(action)
