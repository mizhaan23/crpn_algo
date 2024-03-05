import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    np.random.seed(1234)
    params = np.random.randn(layer.weight.data.numel(), ).astype(np.float32)
    layer.weight.data = torch.from_numpy(params.reshape(layer.weight.data.shape, order="C"))
    return layer


class CategoricalLinearPolicy(nn.Module):
    def __init__(
            self,
            envs,
            init_seed=None,
    ):
        super(CategoricalLinearPolicy, self).__init__()

        input_dim = np.prod(envs.single_observation_space.shape)
        output_dim = envs.single_action_space.n

        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.logits = layer_init(nn.Linear(input_dim, output_dim, bias=False))

    def get_action(self, x):
        action_prob = F.softmax(self.logits(x), dim=-1)

        dist = Categorical(action_prob)
        action = dist.sample()

        return action, dist.log_prob(action)  # action, log_prob
