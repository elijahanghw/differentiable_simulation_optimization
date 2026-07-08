from .bptt import train as train_bptt
from .evolution import train as train_evolution

ALGO_REGISTRY = {
    "bptt": train_bptt,
    "evolution": train_evolution,
}
