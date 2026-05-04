from .apg import train as train_apg

ALGO_REGISTRY = {
    "apg": train_apg,
}
