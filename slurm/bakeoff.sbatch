#!/bin/bash
#SBATCH --job-name=herm-bakeoff
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bakeoff_%j.out

# Model bake-off (handoff §8 q2). Four candidates, one GPU each, same node, same bābs,
# same seed. Each tests a different hypothesis about what this task actually needs:
#
#   Falcon-H1-Arabic-34B-Instruct  Arabic-native beats general-frontier on classical prose.
#                                  Note this is the Arabic line, NOT tiiuae/Falcon-H1-34B-Instruct,
#                                  which is also staged and is the general model.
#   Qwen3.8-27B                    general-frontier wins, because diffing horizons is reasoning.
#                                  Dense, so latency is predictable across the sequential loop.
#   Qwen3.6-35B-A3B                the cheap floor is good enough. 3B active, so several times
#                                  faster per sequential call. If it ties, the V4 duplicate run
#                                  and 3x repetition both become affordable - the highest-value
#                                  single result here.
#   gemma-4-31B-it                 the incumbent. NOT a control that can be skipped: without it
#                                  there is no counterfactual for "we switched", and the paper
#                                  cannot say the switch earned anything. It is also the model
#                                  that produced the first (corrupted) study notes - but under
#                                  3000-char truncation, 300-token caps and free-text prompts,
#                                  none of which still exist. It gets a fair re-test.
#
# Falcon is already in the shared cache; the two Qwens are not. Run
#   bash slurm/stage_models.sh
# on the LOGIN node first - compute nodes are offline, and require_models below will
# refuse the job in seconds rather than burning the allocation.
#
# aya-expanse-32b is the obvious fifth if these four disagree (CC-BY-NC: fine for academic
# use, but say so in the paper).
#
# The run decides two things at once: quality from grounding_rate (free, no extra calls,
# blind rating only as tiebreak) and throughput from the trace, which replaces the
# unverified 2.5 s/call assumption the whole schedule in spec §6.1 rests on.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs/bakeoff

# Client venv for run_study_graph.py (langgraph + langchain-openai; no torch/vllm). The
# vLLM servers activate their own venv inside serve_model - this is only for the parent
# process that drives them. Build once: python -m venv .venv && .venv/bin/pip install -e .
POC_VENV="${POC_VENV:-.venv}"
[[ -f "$POC_VENV/bin/activate" ]] && source "$POC_VENV/bin/activate"

source slurm/_vllm_common.sh

CANDIDATES=(
  "Falcon-H1-Arabic-34B-Instruct"
  "Qwen/Qwen3.8-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "gemma-4-31B-it"
)
N_BABS="${N_BABS:-20}"
MAX_LEN="${VLLM_MAX_LEN:-32768}"
export VLLM_API_KEY=dummy

require_models "${CANDIDATES[@]}"

pids=()
for i in "${!CANDIDATES[@]}"; do
  CUDA_VISIBLE_DEVICES=$i serve_model "$(resolve_model "${CANDIDATES[$i]}")" \
    $((8000 + i)) 1 "$MAX_LEN" > "logs/vllm_bakeoff_$i.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

for i in "${!CANDIDATES[@]}"; do
  wait_for_server $((8000 + i)) "${pids[$i]}"
  echo "ready: ${CANDIDATES[$i]} on port $((8000 + i))"
done

for i in "${!CANDIDATES[@]}"; do
  slug=$(echo "${CANDIDATES[$i]}" | tr '/' '_')
  export VLLM_BASE_URL="http://localhost:$((8000 + i))/v1"
  export VLLM_MODEL_ID="$(resolve_model "${CANDIDATES[$i]}")"
  out="outputs/bakeoff/$slug"
  mkdir -p "$out"
  echo "=== $slug ==="
  python scripts/run_study_graph.py nawawi \
    --limit "$N_BABS" --run-id "$slug" --notes-dir "$out" --auto-approve
done

python scripts/score_bakeoff.py nawawi --bakeoff-dir outputs/bakeoff \
  --sequential-calls 3969 | tee outputs/bakeoff/summary.txt
