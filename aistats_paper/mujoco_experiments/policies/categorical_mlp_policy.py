import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMLPPolicy(nn.Module):
    def __init__(
            self,
            envs,
            hidden_sizes=(32, 32),
            init_std=1.0,
            min_std=1e-6,
            init_seed=None,
    ):
        super(CategoricalMLPPolicy, self).__init__()

        assert len(hidden_sizes) > 0

        input_dim = np.prod(envs.single_observation_space.shape)
        output_dim = envs.single_action_space.n

        if init_seed is not None:
            torch.manual_seed(init_seed)

        input_layer = layer_init(nn.Linear(input_dim, hidden_sizes[0]))
        output_layer = layer_init(nn.Linear(hidden_sizes[-1], output_dim))

        hidden_layers = []
        for i in range(1, len(hidden_sizes)):
            layer = layer_init(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            hidden_layers += [nn.Softmax(dim=-1), layer]

        layers = [input_layer] + hidden_layers + [nn.Softmax(dim=-1), output_layer]

        self.logits = nn.Sequential(*layers)

    def get_action(self, x):
        action_prob = F.softmax(self.logits(x), dim=-1)

        dist = Categorical(action_prob)
        action = dist.sample()

        return action, dist.log_prob(action)  # action, log_prob
