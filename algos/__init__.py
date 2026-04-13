from .apg import train as train_apg
from .ppo import train as train_ppo

ALGO_REGISTRY = {
    "apg": train_apg,
    "ppo": train_ppo,
}
