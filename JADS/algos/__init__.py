from .bptt import train as train_bptt
from .ppo import train as train_ppo

ALGO_REGISTRY = {
    "bptt": train_bptt,
    "ppo": train_ppo,
}
