import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GaussianMLPPolicy(nn.Module):
    def __init__(
            self,
            envs,
            hidden_sizes=(32, 32),
            init_std=1.0,
            min_std=1e-6,
            init_seed=None,
    ):
        super(GaussianMLPPolicy, self).__init__()

        assert len(hidden_sizes) > 0

        self.init_std = torch.tensor(init_std)
        self.min_std = torch.tensor(min_std)

        input_dim = np.prod(envs.single_observation_space.shape)
        output_dim = np.prod(envs.single_action_space.shape)

        if init_seed is not None:
            torch.manual_seed(init_seed)

        input_layer = layer_init(nn.Linear(input_dim, hidden_sizes[0]))
        output_layer = layer_init(nn.Linear(hidden_sizes[-1], output_dim))

        hidden_layers = []
        for i in range(1, len(hidden_sizes)):
            layer = layer_init(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            hidden_layers += [nn.Softmax(dim=-1), layer]

        layers = [input_layer] + hidden_layers + [nn.Softmax(dim=-1), output_layer]

        self.mean_network = nn.Sequential(*layers)
        self.std_network = copy.deepcopy(self.mean_network)


    def get_action(self, x):
        action_mean = F.tanh(self.mean_network(x))
        action_logstd = self.std_network(x)

        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, torch.maximum(action_std, self.min_std))
        action = dist.sample()

        return action, dist.log_prob(action).sum(1)  # action, log_prob
