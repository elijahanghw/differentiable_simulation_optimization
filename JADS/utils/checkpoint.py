import pickle
from pathlib import Path


def save(path: str, policy_params, morph_params=None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"policy_params": policy_params, "morph_params": morph_params}, f)
    print(f"Checkpoint saved to {path}")


def load(path: str) -> tuple:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["policy_params"], ckpt["morph_params"]
