#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vllm_%j.out

# Production serving for the pinned model. See spec §6.
#
# vLLM runs from a venv here (VLLM_VENV, default $PROJECT/Tracing-Mutazila/.venv), and every
# candidate is already staged under $PROJECT/hf_home/hub, so this job downloads nothing and
# runs offline. slurm/_vllm_common.sh does the rest.
#
# H100 94GB is the default because the candidates are 9-34B models that want up to ~70GB
# in bf16 and still need KV cache for a 128K window. The old grete:shared request could
# land on a 40GB A100, where Falcon-H1-34B does not fit. For tensor parallel set VLLM_TP
# and raise --gres to match.
#
# max-model-len is 32768, lowered from 131072 on 2026-08-30. The old comment here said
# "consolidation feeds a whole kitāb into one call (defect #3)" — that has been false
# since defect #3 was fixed. consolidate_kitab chunks its record list at max_chunk_tokens
# (6000) and says so in its own docstring; read_commentary chunks the same way; and
# consolidate_global reduces over per-kitāb summaries. No call in the study graph can
# emit a prompt near even 32k.
#
# Whole kitābs were never representable anyway: measured, they are 222k (Nawawī), 661k
# (ʿAynī) and 751k (Māzandarānī) tokens. There is no context length on this cluster at
# which "one kitāb, one call" works, so raising this is never the answer to a
# consolidation-size problem. At 131072 vLLM wants 18.28 GiB of KV cache and only
# 14.77 GiB is left after weights (job 15637631).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/_vllm_common.sh

MODEL_ID="${VLLM_MODEL_ID:?set VLLM_MODEL_ID}"
require_models "$MODEL_ID"
RESOLVED="$(resolve_model "$MODEL_ID")"

echo "Serving $MODEL_ID -> $RESOLVED on $(hostname):${VLLM_PORT:-8000}"
serve_model "$RESOLVED" "${VLLM_PORT:-8000}" "${VLLM_TP:-1}" "${VLLM_MAX_LEN:-32768}"
