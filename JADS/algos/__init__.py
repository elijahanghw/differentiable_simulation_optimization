from .bptt import train as train_bptt

ALGO_REGISTRY = {
    "bptt": train_bptt,
}
