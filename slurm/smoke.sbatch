#!/bin/bash
#SBATCH --job-name=herm-smoke
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke_%j.out

# One model, one bāb. Proves what the bake-off assumes and nothing has yet shown: that a
# checkpoint actually loads and generates inside the Apptainer container.
#
# Run the free check first, on the LOGIN node - it needs no allocation and catches an
# unrecognised architecture in seconds. Run it from poc/, the same directory you
# sbatch this file from:
#   module load apptainer/1.4.3
#   P=$(readlink -f "$PROJECT")   # $PROJECT is a home-side symlink apptainer cannot see
#   apptainer exec --bind /mnt/vast-kisski:/mnt/vast-kisski --pwd "$(readlink -f "$PWD")" \
#       "$P/Colpali/containers/python-3.11-bookworm.sif" \
#       bash -lc "source $P/Colpali/.venv-vllm-container/bin/activate \
#                 && export HF_HOME=$P/hf_home && python slurm/check_model_configs.py"
#
# Defaults to Falcon-H1-Arabic, the one candidate the abandoned host venv could also serve,
# so a failure here is the container's doing rather than the model's. Override to smoke a
# different candidate - gemma-4-31B-it is the interesting one, since it is what the host
# venv could NOT serve:
#   sbatch --export=ALL,SMOKE_MODEL=gemma-4-31B-it slurm/smoke.sbatch
#
# 45 minutes is deliberate: a 34B load plus one bāb is minutes, so the wall clock is a
# backstop, not a budget. One GPU for well under an hour, against 1067 core-hours for the
# last failed bake-off.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/smoke

POC_VENV="${POC_VENV:-.venv}"
[[ -f "$POC_VENV/bin/activate" ]] && source "$POC_VENV/bin/activate"

source slurm/_vllm_common.sh

SMOKE_MODEL="${SMOKE_MODEL:-Falcon-H1-Arabic-34B-Instruct}"
N_BABS="${N_BABS:-1}"
MAX_LEN="${VLLM_MAX_LEN:-32768}"
export VLLM_API_KEY=dummy

require_models "$SMOKE_MODEL"

serve_model "$(resolve_model "$SMOKE_MODEL")" 8000 1 "$MAX_LEN" > logs/vllm_smoke.log 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

wait_for_server 8000 "$pid" 180 "$RESOLVED"
echo "ready: $SMOKE_MODEL"

# A generation before the graph runs: if the server answers this, loading and decoding both
# work, and anything that fails afterwards belongs to the pipeline rather than the venv.
curl -sf http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-dummy}" \
  -d "{\"model\": \"$(resolve_model "$SMOKE_MODEL")\", \"prompt\": \"بسم\", \"max_tokens\": 8}" \
  || { echo "server is up but will not generate" >&2; exit 1; }
echo

slug=$(echo "$SMOKE_MODEL" | tr '/' '_')
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_ID="$(resolve_model "$SMOKE_MODEL")"

SECONDS=0
python scripts/run_study_graph.py nawawi \
  --limit "$N_BABS" --run-id "smoke_$slug" --notes-dir "outputs/smoke/$slug" --auto-approve
echo "study graph on $N_BABS bāb(s): ${SECONDS}s"
