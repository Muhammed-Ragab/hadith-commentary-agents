#!/bin/bash
#SBATCH --job-name=herm-batch
#SBATCH --partition=grete:shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/batch_%A_%a.out

# Spec §6.2 phase 1. Segmentation and naive readings carry NO cross-unit state,
# so any split across array tasks is valid. ~14,600 calls total.
#
# Submit:  sbatch --export=ALL,COMMENTATOR=nawawi slurm/batch_phase.sbatch
# Merge:   python scripts/run_naive_batch.py nawawi --merge

set -euo pipefail
source "$HOME/.venvs/hermeneutics/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

COMMENTATOR="${COMMENTATOR:?set COMMENTATOR=nawawi|cayni}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
SHARD="${SLURM_ARRAY_TASK_ID}"

python scripts/run_segmentation.py "$COMMENTATOR" \
  --shard "$SHARD" --n-shards "$N_SHARDS"

python scripts/run_naive_batch.py "$COMMENTATOR" \
  --shard "$SHARD" --n-shards "$N_SHARDS"
