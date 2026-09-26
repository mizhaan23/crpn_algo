
import flax.linen as nn
from distrax import Categorical

class CategoricalLinearPolicy(nn.Module):
    output_dim: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim, use_bias=False)(x)

    def get_action(self, params, x, key):
        dist = Categorical(logits=self.apply(params, x))
        action = dist.sample(seed=key)
        return action, dist.log_prob(action)

    def get_log_prob(self, params, x, action):
        dist = Categorical(logits=self.apply(params, x))
        return dist.log_prob(action)

