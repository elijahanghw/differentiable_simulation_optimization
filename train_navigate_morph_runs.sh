#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/navigate_morph.yaml"
LOG="logs/navigate_morph_logs.csv"
CKPT="checkpoints/navigate_morph.pkl"
LOG_DIR="logs/navigate_morph"
CKPT_DIR="checkpoints/navigate_morph"
N_RUNS=10

mkdir -p "$LOG_DIR" "$CKPT_DIR"

for i in $(seq 1 $N_RUNS); do
    echo "========== Run $i / $N_RUNS =========="
    python train.py --config "$CONFIG"
    mv "$LOG"  "$LOG_DIR/navigate_morph_logs_${i}.csv"
    mv "$CKPT" "$CKPT_DIR/navigate_morph_${i}.pkl"
    echo "Saved run $i artifacts."
done

echo "All $N_RUNS runs complete."
