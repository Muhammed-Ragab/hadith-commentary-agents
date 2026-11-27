#!/bin/bash
# Pre-stage bake-off weights into the shared cache. RUN ON THE LOGIN NODE.
#
# Compute nodes run offline (HF_HUB_OFFLINE=1) against $PROJECT/hf_home, so anything not
# already in the cache has to be fetched here first. slurm/_vllm_common.sh will refuse to
# submit a job naming a model that is missing, which is what this script exists to fix.
#
#   bash slurm/stage_models.sh            # stage everything the bake-off needs
#   bash slurm/stage_models.sh <repo-id>  # stage one extra model

set -euo pipefail
export HF_HOME="${HF_HOME:-$PROJECT/hf_home}"

DEFAULT_MODELS=(
  "Qwen/Qwen3.8-27B"
  "Qwen/Qwen3.6-35B-A3B"
)
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

# Falcon-H1-Arabic-34B-Instruct and gemma-4-31B-it are already staged; nothing to do there.
for id in "${MODELS[@]}"; do
  echo "=== $id ==="
  hf download "$id" --quiet || huggingface-cli download "$id" --quiet
done

echo
echo "Staged under $HF_HOME/hub. Verify before submitting:"
echo "  ls \$HF_HOME/hub | grep -i qwen3"
