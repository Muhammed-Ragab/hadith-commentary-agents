#!/bin/bash
#SBATCH --job-name=herm-matched
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matched_%j.out

# Matched-sample study run: every commentator against ONE model load.
#
# Why one job and not serve_vllm.sbatch + three study_phase.sbatch jobs: serve_model
# never passes --host, so vLLM binds 127.0.0.1 and is unreachable from another node.
# Serving inside the job is what smoke.sbatch already does, and it also means the three
# arms share a single model load instead of paying it three times.
#
# Why one job and not three: comparability is the claim. Same weights, same server,
# same prompts, same bāb budget across arms -- so a difference between commentators is
# a difference in the commentary, not in the run.
#
#   sbatch slurm/study_matched.sbatch
#   sbatch --export=ALL,N_BABS=50,STUDY_MODEL=gemma-4-31B-it slurm/study_matched.sbatch
#   sbatch --partition=grete:shared --gres=gpu:A100:1 --time=03:00:00 \
#     --export=ALL,N_BABS=5,STUDY_MODEL=ALLaM-7B-Instruct-preview,VLLM_MAX_LEN=4096,STUDY_CHUNK_TOKENS=2000,SEGMENTATION_MAX_CHARS=6000,STUDY_MAX_OUTPUT_TOKENS=800,STUDY_CONSOLIDATE_MAX_TOKENS=1000 \
#     slurm/study_matched.sbatch
#
# --auto-approve is on: this is a measurement pass. Drop it for a citable run and take
# the kitāb interrupts, since the scholar in the loop is part of the design (spec §5).

set -euo pipefail
# STUDY_ENDPOINT names an OpenAI-compatible server that is already running somewhere else
# -- GWDG SAIA, say, which serves qwen3.8-27b without a GPU allocation. Everything past
# the server block is backend-agnostic already, so the whole pipeline runs off-cluster
# with no second copy of it to keep in sync. Unset (the normal case): serve vLLM here.
#   STUDY_ENDPOINT=https://chat-ai.academiccloud.de/v1 STUDY_API_KEY="$SAIA_API_KEY" \
#     STUDY_MODEL=qwen3.8-27b N_BABS=2 bash slurm/study_matched.sbatch
EXTERNAL="${STUDY_ENDPOINT:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/method_files_matched

POC_VENV="${POC_VENV:-.venv}"
[[ -f "$POC_VENV/bin/activate" ]] && source "$POC_VENV/bin/activate"

if [[ -z "$EXTERNAL" ]]; then
  # _vllm_common.sh serves from the apptainer container (vllm 0.21.0). _vllm_venv.sh serves
  # from a host venv and is what Qwen3.8-27B needs -- see that file's header, and §8.15.
  source "slurm/${VLLM_LAUNCHER:-_vllm_common.sh}"
fi

STUDY_MODEL="${STUDY_MODEL:-Falcon-H1-Arabic-34B-Instruct}"
N_BABS="${N_BABS:-25}"
DEFAULT_MAX_LEN=32768
[[ "$STUDY_MODEL" == Falcon-H1-Arabic-34B-Instruct ]] && DEFAULT_MAX_LEN=28672
MAX_LEN="${VLLM_MAX_LEN:-$DEFAULT_MAX_LEN}"
COMMENTATORS="${COMMENTATORS:-nawawi cayni mazandarani}"
export VLLM_API_KEY="${STUDY_API_KEY:-dummy}"

if [[ -n "$EXTERNAL" ]]; then
  # Nothing to stage or resolve: the endpoint publishes its own model names. SLURM_JOB_ID
  # does not exist off-cluster, but every output path and log name is keyed on it.
  RESOLVED="$STUDY_MODEL"
  : "${SLURM_JOB_ID:=api$(date +%Y%m%d%H%M%S)}"
else
  require_models "$STUDY_MODEL"
  RESOLVED="$(resolve_model "$STUDY_MODEL")"
fi

if [[ -n "$EXTERNAL" ]]; then
  export VLLM_BASE_URL="$EXTERNAL"
  echo "using $STUDY_MODEL at $EXTERNAL"
else
server_args=()
# Measured 7,697 s/bab against Falcon's 950 on the container stack, and the parser is the
# suspect: it makes the model emit chain-of-thought before every JSON answer. It may not
# even apply once vllm 0.26 registers these weights as Qwen3_5ForConditionalGeneration.
# VLLM_REASONING_PARSER=none turns it off so the two can be measured against each other.
RP="${VLLM_REASONING_PARSER:-auto}"
[[ "$RP" == auto && "$STUDY_MODEL" == Qwen/Qwen3.* ]] && RP=qwen3
[[ "$RP" != auto && "$RP" != none ]] && server_args+=(--reasoning-parser "$RP")
serve_model "$RESOLVED" 8000 "${VLLM_TP:-1}" "$MAX_LEN" "${server_args[@]}" > logs/vllm_matched_"$SLURM_JOB_ID".log 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

wait_for_server 8000 "$pid" 180 "$RESOLVED"
echo "ready: $STUDY_MODEL -> $RESOLVED"

curl -sf http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${VLLM_API_KEY:-dummy}" \
  -d "{\"model\": \"$RESOLVED\", \"prompt\": \"بسم\", \"max_tokens\": 8}" \
  || { echo "server is up but will not generate" >&2; exit 1; }
echo
export VLLM_BASE_URL="http://localhost:8000/v1"
fi

slug=$(echo "$STUDY_MODEL" | tr '/' '_')
export VLLM_MODEL_ID="$RESOLVED"
# RUN_TAG is the checkpoint identity. run_study_graph puts its SQLite checkpointer under
# outputs/runs/$run_tag_$arm, so a tag containing $SLURM_JOB_ID gives every job a fresh
# checkpointer and makes resume impossible -- which is why the 123 GPU-hour cayni arm could
# not be split across 48h allocations. Pass the SAME RUN_TAG to chain jobs; leave it unset
# for a one-shot run, where per-job isolation is what you want.
run_tag="${RUN_TAG:-matched_${slug}_${SLURM_JOB_ID}}"

# STUDY_SAMPLE selects a stratified sample (KITABS:PER_KITAB:SEED) instead of N_BABS'
# first N babs, which only ever sampled the opening kitabs -- and 8.17 showed the
# opening is the least representative part of the corpus.
#
# Per-arm override: STUDY_SAMPLE_NAWAWI / _CAYNI / _MAZANDARANI. The arms need different
# specs because their kitab counts differ by an order of magnitude -- Mazandarani has 5
# bab-bearing kitabs (is_bab drops the undivided ones) against Cayni's 87 -- so one
# KITABS:PER_KITAB cannot give all three a comparable N. Measured 2026-09-14:
#   STUDY_SAMPLE_NAWAWI=14:5:20260914       -> 65 babs / 14 kitabs
#   STUDY_SAMPLE_CAYNI=12:5:20260914        -> 60 babs / 12 kitabs
#   STUDY_SAMPLE_MAZANDARANI=5:12:20260914  -> 60 babs /  5 kitabs
#
# Resolved once here and handed to every phase verbatim: segmentation, the naive
# pre-pass, the study graph and the validator all join on bab_ref, so a phase given a
# different spec silently unmatches the arms rather than failing.
arm_sample() {
  local c="$1" var="STUDY_SAMPLE_${1^^}"
  printf '%s' "${!var:-${STUDY_SAMPLE:-}}"
}

validate_args=(--expected-babs "$N_BABS")
if [[ -n "${STUDY_SAMPLE:-}" || -n "${STUDY_SAMPLE_NAWAWI:-}${STUDY_SAMPLE_CAYNI:-}${STUDY_SAMPLE_MAZANDARANI:-}" ]]; then
  validate_args=()
  [[ -n "${STUDY_SAMPLE:-}" ]] && validate_args+=(--sample "$STUDY_SAMPLE")
  for c in $COMMENTATORS; do
    spec="$(arm_sample "$c")"
    [[ -n "$spec" && "$spec" != "${STUDY_SAMPLE:-}" ]] && validate_args+=(--sample "${c}=${spec}")
  done
fi

arm_select_args() {
  local spec
  spec="$(arm_sample "$1")"
  if [[ -n "$spec" ]]; then
    printf '%s\n' --sample "$spec"
  else
    printf '%s\n' --limit "$N_BABS"
  fi
}

out_root="outputs/method_files_matched/$run_tag"
seg_dir="$out_root/segmentation"
mkdir -p "$out_root" "$seg_dir"

# The naive reading must not see the method hypothesis, so it is a separate stateless pass
# by design (spec §6.2 phase 1b). --live-naive computed it inline instead, which put ~a
# third of the per-bāb calls on the sequential path for no methodological gain. STUDY_NAIVE
# picks: "batch" precomputes it up front (concurrent across arms, no state to serialise),
# "live" restores the old inline behaviour.
naive_args=()
if [[ "${STUDY_NAIVE:-batch}" == live ]]; then
  naive_args+=(--live-naive)
else
  echo "===== naive pre-pass ====="
  npids=()
  for c in $COMMENTATORS; do
    ( mapfile -t sel < <(arm_select_args "$c"); python scripts/run_naive_batch.py "$c" "${sel[@]}" --run-id "${run_tag}_${c}"       && python scripts/run_naive_batch.py "$c" --merge --run-id "${run_tag}_${c}" ) > "logs/naive_${SLURM_JOB_ID}_${c}.log" 2>&1 &
    npids+=($!)
  done
  nfail=0
  for i in "${!npids[@]}"; do wait "${npids[$i]}" || nfail=1; done
  for c in $COMMENTATORS; do cat "logs/naive_${SLURM_JOB_ID}_${c}.log"; done
  [[ $nfail -eq 0 ]] || { echo "naive pre-pass failed; rerun with STUDY_NAIVE=live" >&2; exit 1; }
fi

# The three arms share the model load but are otherwise independent -- separate corpora,
# separate checkpointers, separate method hypotheses. Running them serially left vLLM at
# "Maximum concurrency 1.00x" with one request in flight; running them together gives it
# three streams to batch and bounds the job by the longest arm instead of their sum.
#
# How much that actually buys depends on KV headroom, and Falcon's is thin (4.17 GiB at
# 28,672 tokens). vLLM queues rather than fails when three long requests do not fit, so
# this is safe either way -- but measure before assuming the arms really overlap.
# COMMENTATORS=nawawi restores single-arm behaviour.
#
# A twice-invalid bāb (segmentation that never finds a credible boundary, or a diff
# horizon_diff gives up on twice) is logged and the graph continues rather than aborting
# the arm (B4) -- validate_matched.py then requires every requested bāb to have either a
# note or a matching logged failure, and compare_methods.py prints the shortfall by name.
# Keep the outer status guard here for transport/process failures, which are a different
# thing: those never get to log anything.
run_arm() {
  local c="$1"
  SECONDS=0
  mapfile -t sel < <(arm_select_args "$c")
  if python scripts/run_segmentation.py "$c"         "${sel[@]}"         --run-id "${run_tag}_${c}"         --output-dir "$seg_dir"     && python scripts/run_study_graph.py "$c"         "${sel[@]}"         --run-id "${run_tag}_${c}"         --segmentation-dir "$seg_dir"         "${naive_args[@]}"         --notes-dir "$out_root/$c"         --auto-approve; then
    echo "$c: ok in ${SECONDS}s"
  else
    echo "$c: FAILED after ${SECONDS}s" >&2
    return 1
  fi
}

status=0
arm_pids=(); arm_names=()
for c in $COMMENTATORS; do
  run_arm "$c" > "logs/matched_${SLURM_JOB_ID}_${c}.log" 2>&1 &
  arm_pids+=($!); arm_names+=("$c")
done
for i in "${!arm_pids[@]}"; do
  wait "${arm_pids[$i]}" || status=1
done
# Concurrent arms interleave stdout, so each writes its own log and they are replayed in a
# fixed order here. The job log stays readable and comparable across runs.
for c in $COMMENTATORS; do
  echo "===== $c ====="
  cat "logs/matched_${SLURM_JOB_ID}_${c}.log"
done

echo
echo "notes under $out_root"
python scripts/compare_methods.py "$out_root" || status=1
python scripts/validate_matched.py "$out_root" "${validate_args[@]}" || status=1
exit "$status"
