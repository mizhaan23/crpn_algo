from typing import Callable, Any, Tuple, NamedTuple
from functools import partial

from . import common
from . import reinforce
from . import acrpn


class Optimizer(NamedTuple):
    name: str
    init: Callable[[Any], Any]
    step: Callable[[Any, Any, Callable, Callable, int], Tuple[Any, Any]]

    def __iter__(self):
        return iter((self.init, self.step))


def get_optimizer(name: str, **default_kwargs) -> Optimizer:
    """
    Factory function returning an Optimizer instance with .init and .step methods.
    
    Supported names:
      - 'reinforce': Standard REINFORCE (lr)
      - 'acrpn': Adaptive Cubic Regularization with Hessian (alpha)
    """
    key = name.lower()

    if key == "reinforce":
        step_fn = partial(reinforce.step, **default_kwargs)
        return Optimizer(name=name, init=reinforce.init, step=step_fn)

    elif key == "acrpn":
        kwargs = {"use_hessian": True, **default_kwargs}
        step_fn = partial(acrpn.step, **kwargs)
        return Optimizer(name=name, init=acrpn.init, step=step_fn)

    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Available: ['reinforce', 'acrpn']")


__all__ = ["get_optimizer", "Optimizer", "common", "reinforce", "acrpn"]

