import os
import argparse
from distutils.util import strtobool


def parse_args(default_exp_name="experiment", multi=False):
    """Universal command-line argument parser for RL experiments."""
    parser = argparse.ArgumentParser()

    # Environment specific args
    parser.add_argument("--exp-name", type=str, default=default_exp_name,
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
                        help="the id of the gym environment")
    parser.add_argument("--env-seed", type=int, default=1,
                        help="the seed of the gym environment")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed of all rngs")

    # Algorithm specific args
    parser.add_argument("--alpha", type=float, default=1000.0,
                        help="the regularization parameter of the CRPN algorithm")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate for first-order or normalized updates")
    parser.add_argument("--beta-1", type=float, default=0.0,
                        help="momentum coefficient for policy gradients (default: 0.0)")
    parser.add_argument("--beta-2", type=float, default=0.0,
                        help="momentum coefficient for Hessian curvature (default: 0.0)")
    parser.add_argument("--krylov-dim", type=int, default=3,
                        help="dimension k of the Krylov subspace for Lanczos submodel solver (default: 3)")
    parser.add_argument("--use-hessian", type=lambda x: bool(strtobool(x)), default=True,
                        help="whether to use the second-order Hessian term")
    parser.add_argument("--normalize-returns", type=lambda x: bool(strtobool(x)), default=True,
                        help="will normalize the returns by standard scaling")
    parser.add_argument("--activation", type=str, default="relu", choices=["tanh", "relu", "softmax"],
                        help="activation function for MLP policies: 'tanh', 'relu', or 'softmax'")
    parser.add_argument("--hidden-sizes", type=eval, default=(),
                        help="hidden-sizes of the neural network")
    parser.add_argument("--policy-type", type=str, default=None, choices=["mlp", "conv", None],
                        help="policy architecture type: 'mlp', 'conv', or None (auto-detect)")

    # Simulation specific args
    parser.add_argument("--max-timesteps", type=int, default=500,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="total update epochs for the policy")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="the number of parallel game environments")
    parser.add_argument("--discount-factor", type=float, default=0.99,
                        help="discount factor (gamma) for returns")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be saved locally")

    if multi:
        parser.add_argument("--num-seeds", type=int, default=10,
                            help="number of seeds to run")
        parser.add_argument("--mode", type=str, default="vmap", choices=["vmap", "sequential"],
                            help="run mode: vmap or sequential")
        parser.add_argument("--chunk-size", type=int, default=None,
                            help="chunk size for chunked vmap execution (default: None, full vmap)")

    return parser.parse_args()
