from .pendulum.pendulum import Pendulum
from .pendulum.pendulum_force import PendulumForce
from .quad_2d.hover import Hover2d
from .multicopter.hover import Hover3d
from .multicopter.navigate import Navigate
from .pointmass.navigate import PointMassNavigate


ENV_REGISTRY = {
    "pendulum": Pendulum,
    "pendulum_force": PendulumForce,
    "hover_2d": Hover2d,
    "hover_3d": Hover3d,
    "navigate": Navigate,
    "pointmass_navigate": PointMassNavigate,
}


def make_env(name: str, **kwargs):
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{name}'. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name](**kwargs)
