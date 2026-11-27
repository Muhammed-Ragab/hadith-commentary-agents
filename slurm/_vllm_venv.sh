#!/bin/bash
# Host-venv vLLM launcher. Drop-in replacement for _vllm_common.sh's serve_model.
#
# Why this exists: the container venv (_vllm_common.sh) reports vllm 0.21.0, which
# registers Qwen3.8-27B as a Qwen3-Next model and routes its prefill through
# FlashInfer's gated-delta-net kernel. That kernel JIT-compiles at first forward pass
# and needs nvcc, which no KISSKI compute node has -- job 15789374 died there in five
# minutes. There is no env var to disable that path the way
# VLLM_USE_FLASHINFER_SAMPLER=0 disables the sampler one.
#
# vllm 0.26.0 registers the same weights as Qwen3_5ForConditionalGeneration and never
# reaches the GDN kernel. The EMNLP project (kisski-futurmig/edu-vlm/testruns) has been
# running exactly that on these nodes since 2026-09-08; its install recipe lives in
# training/slurm/teacher_qwen38.sh and its verified stack is:
#
#   vllm 0.26.0 | torch 2.11.0+cu128 | CUDA 12.8 | H100 80 GB (kisski-h100)
#
# The glibc objection in _vllm_common.sh's header does not apply to those wheels:
# manylinux_2_28 targets glibc 2.28, which is what Rocky 8 ships.
#
# Only serve_model changes. resolve_model, require_models and wait_for_server come from
# _vllm_common.sh unchanged, so a fix to any of them lands once.
#
# Usage -- point VLLM_HOST_VENV at a venv holding vllm >= 0.26:
#   sbatch --export=ALL,VLLM_LAUNCHER=_vllm_venv.sh,\
#VLLM_HOST_VENV=/mnt/vast-kisski/projects/kisski-futurmig/edu-vlm/testruns/env-vllm,\
#STUDY_MODEL=Qwen/Qwen3.8-27B,N_BABS=25 \
#     --time=06:00:00 slurm/study_matched.sbatch   # submit from poc/, paths are relative to SLURM_SUBMIT_DIR
#
# Borrowing EMNLP's env-vllm is the zero-setup way to test. Build our own before relying
# on it for a citable run -- it is another project's environment and can change without
# notice. The recipe is four pip commands, run from the LOGIN node (compute nodes have
# no internet) into a venv on shared VAST so the GPU node sees it.

source "$(dirname "${BASH_SOURCE[0]}")/_vllm_common.sh"

VLLM_HOST_VENV="$(readlink -f "${VLLM_HOST_VENV:-$PROJECT/edu-vlm/testruns/env-vllm}" 2>/dev/null)"

# serve_model <resolved-model> <port> <tensor-parallel> <max-model-len> [extra vllm args...]
# Same contract as the container version: call backgrounded (&), then wait_for_server.
serve_model() {
  local model="$1" port="$2" tp="$3" maxlen="$4"; shift 4

  [[ -n "$VLLM_HOST_VENV" && -f "$VLLM_HOST_VENV/bin/activate" ]] || {
    echo "no host venv at ${VLLM_HOST_VENV:-<unset>} -- set VLLM_HOST_VENV, see this file's header" >&2
    exit 1
  }

  # Same reasoning as the container path: the study graph is single-stream, so capacity
  # for 512 concurrent sequences is waste and the cudagraph capture that sizes it is what
  # OOM'd jobs 15518082/15557225/15637520.
  local eager=""
  [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]] && eager="--enforce-eager"

  # These two are about the NODE, not the container -- there is no nvcc on either, so any
  # JIT-compiled kernel extension fails in both. Keep them.
  #
  # VLLM_WORKER_MULTIPROC_METHOD=spawn is specific to the host route: vLLM V1 forks its
  # EngineCore, something in the parent touches CUDA first, and the fork dies with
  # "Cannot re-initialize CUDA in forked subprocess" (EMNLP job 15638726).
  #
  # env -u VIRTUAL_ENV: study_matched.sbatch has already activated the pipeline venv, and
  # stacking a second activate over it leaves PATH pointing at the wrong python.
  env -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH bash -lc "
    source '$VLLM_HOST_VENV/bin/activate'
    export HF_HOME='$HF_HOME' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export TOKENIZERS_PARALLELISM=false
    export VLLM_API_KEY='${VLLM_API_KEY:-dummy}'
    export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
    python -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"| torch\", torch.__version__)' >&2
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
