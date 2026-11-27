#!/bin/bash
# Shared vLLM launch helpers for the KISSKI setup.
#
# vLLM runs INSIDE APPTAINER, from a venv built inside the container. Not from a host venv.
# This is the setup Colpali established and documented in its kisski_vllm_runbook.md, and
# it is required rather than preferred: KISSKI compute nodes are Rocky 8 with glibc 2.28 and
# a driver exposing CUDA 12.8, while current vLLM wheels want newer glibc and link
# libcudart.so.13. The container supplies glibc 2.36, --nv injects the host driver, and the
# combination is the only one that satisfies all three constraints at once.
#
# The host venv route was tried and abandoned ONCE, for a reason that was about the venv and
# not the route: $PROJECT/Tracing-Mutazila/.venv carries vllm 0.11.2, whose registry has no
# Gemma4ForConditionalGeneration, Qwen3_5ForConditionalGeneration or
# Qwen3_5MoeForConditionalGeneration - three of the four bake-off candidates. Job 15479179
# died on it and cost 1067 core-hours. The container venv reports vllm 0.21.0, which has
# Gemma 4 but routes Qwen3.8-27B through the Qwen3-Next GDN path and its nvcc-only FlashInfer
# kernel (job 15789374).
#
# A host venv DOES work with the right wheels: the EMNLP project runs vllm 0.26.0 /
# torch 2.11.0+cu128 from $PROJECT/edu-vlm/testruns/env-vllm on these same nodes, where
# Qwen3.8-27B registers as Qwen3_5ForConditionalGeneration and never reaches the GDN kernel.
# manylinux_2_28 wheels target glibc 2.28, which is exactly Rocky 8. See _vllm_venv.sh.
#
# Every model is pre-staged under $PROJECT/hf_home/hub, so jobs run fully offline: nothing
# is downloaded, and a missing model fails at submit time rather than 40 minutes into an
# allocation.
#
# Override before sourcing if your paths differ:
#   VLLM_SIF      container image (default: Colpali's python:3.11-bookworm pull)
#   VLLM_VENV     venv built INSIDE that container (default: Colpali's .venv-vllm-container)
#   APPTAINER_MOD module to load for apptainer
#   HF_HOME       defaults to $PROJECT/hf_home

# Canonical, for the same reason as VLLM_SIF/VLLM_VENV below: the container cannot see
# $PROJECT's home-side symlink, and a hub path it cannot stat makes every model look
# unstaged even though the weights are right there.
export HF_HOME="$(readlink -f "${HF_HOME:-$PROJECT/hf_home}" 2>/dev/null || echo "${HF_HOME:-$PROJECT/hf_home}")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
HF_HUB_DIR="$HF_HOME/hub"

# Canonicalised on purpose. $PROJECT is /mnt/vast-kisski/home/<user>/.project/dir.project,
# a symlink to /mnt/vast-kisski/projects/kisski-futurmig. Binding /mnt/vast-kisski does not
# carry the nested mountpoints through, so the home-side path does not exist inside the
# container while the resolved projects-side path does. Every path handed to apptainer
# goes through readlink -f.
VLLM_SIF="$(readlink -f "${VLLM_SIF:-$PROJECT/Colpali/containers/python-3.11-bookworm.sif}" 2>/dev/null)"
VLLM_VENV="$(readlink -f "${VLLM_VENV:-$PROJECT/Colpali/.venv-vllm-container}" 2>/dev/null)"
APPTAINER_MOD="${APPTAINER_MOD:-apptainer/1.4.3}"

# Bind the real paths, not the symlinked ones. /projects/extern/... is a symlink into
# /mnt/vast-kisski, and transformers inside the container treats a path it cannot stat as a
# Hub repo id - the "Repo id must be in the form..." failure in the runbook.
_bind_paths() {
  local binds="/mnt/vast-kisski:/mnt/vast-kisski"
  local p seen=""
  # The venv is included because it usually lives under another project (Colpali), which
  # neither the submit dir nor HF_HOME covers.
  for p in "$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}")" "$(readlink -f "$HF_HOME")" "$VLLM_VENV" "$(dirname "$VLLM_SIF")"; do
    [[ -n "$p" && -d "$p" && ",$seen," != *",$p,"* ]] || continue
    binds="$binds,$p:$p"
    seen="$seen,$p"
  done
  echo "$binds"
}

# The cache mixes two layouts: bare snapshot directories (Falcon-H1-Arabic-34B-Instruct)
# and standard HF cache dirs (models--Qwen--Qwen3.5-27B). Resolve a bare directory to its
# absolute path; otherwise hand vLLM the repo id and let it read the cache.
resolve_model() {
  local id="$1" base="${1##*/}"
  if [[ -d "$HF_HUB_DIR/$base" ]]; then
    echo "$HF_HUB_DIR/$base"
  elif [[ -d "$HF_HUB_DIR/models--${id//\//--}" ]]; then
    echo "$id"
  else
    echo "MISSING" >&2
    return 1
  fi
}

require_models() {
  local missing=0
  for id in "$@"; do
    if ! resolve_model "$id" > /dev/null 2>&1; then
      echo "NOT STAGED: $id  (expected under $HF_HUB_DIR)" >&2
      missing=1
    fi
  done
  [[ $missing -eq 0 ]] || { echo "Stage the missing models before submitting." >&2; exit 1; }
}

# serve_model <resolved-model> <port> <tensor-parallel> <max-model-len> [extra vllm args...]
# Runs vLLM's OpenAI server inside the container. Call backgrounded (&) with
# CUDA_VISIBLE_DEVICES set per model, as bakeoff.sbatch does; apptainer inherits it and
# --nv passes only the visible devices through.
serve_model() {
  local model="$1" port="$2" tp="$3" maxlen="$4"; shift 4
  # Checked here rather than at source time: _vllm_venv.sh sources this file for
  # resolve_model/require_models/wait_for_server and never touches the container.
  [[ -n "$VLLM_SIF" && -f "$VLLM_SIF" ]] || { echo "no container at $VLLM_SIF (see Colpali docs/kisski_vllm_runbook.md)" >&2; exit 1; }
  [[ -n "$VLLM_VENV" && -f "$VLLM_VENV/bin/activate" ]] || { echo "no container venv at $VLLM_VENV" >&2; exit 1; }
  module load "$APPTAINER_MOD" 2>/dev/null || true

  # Falcon-H1 is a hybrid Mamba/attention model. Three jobs (15518082, 15557225,
  # 15637520) died identically in profile_cudagraph_memory -- 1.02 GiB short, ~92 GiB
  # in use -- while capturing CUDA graphs for 51 batch sizes up to 512 and sizing the
  # mamba state cache for that concurrency. That allocation scales with CONCURRENCY,
  # not context or the utilization fraction, which is why --max-model-len 131072->32768
  # and --gpu-memory-utilization 0.92->0.85 both changed nothing.
  #
  # The study graph is single-stream: one request at a time, sequentially. Capacity for
  # 512 concurrent sequences is waste. Set VLLM_ENFORCE_EAGER=0 to re-enable cudagraphs
  # if a throughput measurement needs them and memory allows.
  local eager=""
  [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]] && eager="--enforce-eager"

  # VLLM_USE_FLASHINFER_SAMPLER=0 is load-bearing, not tuning: FlashInfer JIT-builds a CUDA
  # extension during warmup, and the container has the CUDA runtime through --nv but no
  # nvcc. Without it, warmup dies on "Could not find nvcc".
  #
  # VLLM_USE_DEEP_GEMM=0 is the same problem with a different kernel (job 15637845):
  # "Failed to import vendored deep_gemm", then kernel_warmup -> deep_gemm_warmup raises
  # "DeepGEMM backend is not available or outdated". It is an FP8 kernel library and every
  # candidate here is bf16, so nothing is lost by disabling it. Expect this pattern again:
  # any JIT-compiled kernel extension fails in this container, and vLLM's warmup treats
  # an unavailable backend as fatal rather than falling back.
  #
  # gpu-memory-utilization defaults to 0.85, not 0.92: vLLM fills that fraction with
  # weights + KV cache and then profile_cudagraph_memory needs headroom BEYOND it, so
  # 0.92 left 925 MiB free and OOM'd on a 1.02 GiB profiling allocation (jobs 15518082,
  # 15557225). Lowering --max-model-len does NOT help -- it changes how many blocks fit
  # in the budget, not the size of the budget. This flag is the lever; raise VLLM_GPU_UTIL
  # only if KV cache is genuinely short, and never above ~0.90 on a 34B.
  apptainer exec --nv --bind "$(_bind_paths)" "$VLLM_SIF" bash -lc "
    source '$VLLM_VENV/bin/activate'
    export HF_HOME='$HF_HOME' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_API_KEY='${VLLM_API_KEY:-dummy}'
    export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
    exec python -m vllm.entrypoints.openai.api_server \
      --model '$model' \
      --served-model-name '$model' \
      --port '$port' \
      --tensor-parallel-size '$tp' \
      --max-model-len '$maxlen' \
      --seed 0 \
      --gpu-memory-utilization '${VLLM_GPU_UTIL:-0.85}' \
      --max-num-seqs '${VLLM_MAX_SEQS:-16}' \
      $eager $*"
}

# wait_for_server <port> <pid> [tries]
# The pid is not optional: without it a server that dies on an import error still burns
# the full 60-minute poll (job 15479179 cost 1067 core-hours that way).
# serve_model exports VLLM_API_KEY inside the container, so every /v1 route demands a
# bearer token. Polling /v1/models without one returns 401 forever and the job times out
# with a perfectly healthy server behind it (job 15638110, the first run ever to reach
# startup). Send the key, and accept any HTTP response as proof the app is listening --
# a 401 means the server is up, which is all this function needs to know.
# A 4th arg (model id) makes the gate an actual completion instead of /v1/models.
# /v1/models answers from the frontend while the engine is still loading weights, so it
# reports ready mid-load and the first real request fails (job 15781148, killed at shard
# 11/18). Generating one token is the only proof the engine is live.
wait_for_server() {
  local port="$1" pid="$2" tries="${3:-180}" model="${4:-}"
  local key="${VLLM_API_KEY:-dummy}"
  local probe=(curl -sf -H "Authorization: Bearer $key" "http://localhost:$port/v1/models")
  [[ -n "$model" ]] && probe=(curl -sf -H 'Content-Type: application/json'
    -H "Authorization: Bearer $key" "http://localhost:$port/v1/completions"
    -d "{\"model\": \"$model\", \"prompt\": \"بسم\", \"max_tokens\": 1}")
  until "${probe[@]}" > /dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "vLLM for port $port exited during startup; see logs/vllm_*.log" >&2; return 1; }
    ((tries--)) || { echo "server on $port never came up" >&2; return 1; }
    sleep 20
  done
}
