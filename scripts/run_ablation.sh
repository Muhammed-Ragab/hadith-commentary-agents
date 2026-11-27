#!/bin/bash
# Ablation over the two saturation-curve mechanisms measured on run 15789363 (see
# hermeneutics/nodes.py Deps.legacy_example_prompt / legacy_hypothesis_menu and
# hermeneutics/prompts.py's read_commentary / read_commentary_legacy_example).
#
# Bash, not Python: every step here is "run this existing CLI with these flags", the
# same shape as slurm/study_matched.sbatch's off-cluster invocation (which this script
# borrows its STUDY_ENDPOINT convention from) -- no new argument parsing, JSON
# handling, or control flow that would need a real script, just three ordered CLI
# launches sharing one naive-reading pass.
#
# Runs ONE arm (nawawi), a 12-bāb stratified sample, three times against a live
# OpenAI-compatible endpoint (SAIA):
#   A  both mechanisms present  (current, pre-fix behaviour)
#   B  fix 1 only                (example neutralized; hypothesis menu still counted
#                                  and frequency-ranked)
#   C  both fixes                (the new default -- what a bare run_study_graph.py
#                                  invocation already does with no flags)
#
# This needs a live SAIA endpoint and an API key this environment does not have.
# Prepared ready-to-launch; NOT run. Launch with:
#
#   STUDY_ENDPOINT=https://chat-ai.academiccloud.de/v1 \
#   STUDY_API_KEY="$SAIA_API_KEY" \
#   VLLM_MODEL_ID=qwen3.8-27b \
#   bash scripts/run_ablation.sh
#
# Estimated cost: 12 bābs x 3 conditions = 36 bāb-arms. At ~7 min/bāb serial on SAIA
# (the project's standing estimate for this endpoint), that is about 4h12m wall clock
# run as written (the three conditions one after another, so a later one's kitāb
# review does not race an earlier one's). Naive readings are computed once, up front,
# and reused by all three conditions -- naive_reading never sees the method hypothesis
# or the read_commentary prompt, so neither toggle can affect it, and paying for it
# three times would just be 12 wasted calls.
#
# To read the result once the three runs exist:
#   python scripts/compare_methods.py outputs/ablation/A outputs/ablation/B outputs/ablation/C

set -euo pipefail

: "${STUDY_ENDPOINT:?set STUDY_ENDPOINT to a live OpenAI-compatible endpoint (e.g. SAIA)}"
: "${VLLM_MODEL_ID:?set VLLM_MODEL_ID to the model name the endpoint serves}"
export VLLM_BASE_URL="$STUDY_ENDPOINT"
export VLLM_API_KEY="${STUDY_API_KEY:-dummy}"

cd "$(dirname "$0")/.."

COMMENTATOR=nawawi
# 12 kitābs x 1 bāb each = 12 bābs, spread across the corpus rather than its opening
# (spec's stratified sampler, scripts/check_corpus.py:sample_indices). Every phase of
# a run must share this exact spec, or bab_ref positions stop lining up.
SAMPLE="${SAMPLE:-12:1:20260914}"
# ponytail: env-overridable so a cluster run can tag its own output root and not land
# on top of a local run's (e.g. OUT_ROOT=outputs/ablation_$SLURM_JOB_ID).
OUT_ROOT="${OUT_ROOT:-outputs/ablation}"
NAIVE_RUN_ID="${NAIVE_RUN_ID:-ablation_naive}"

mkdir -p "$OUT_ROOT"

SEG_DIR="${OUT_ROOT}/segmentation"
mkdir -p "$SEG_DIR"

# Segmentation, once, shared by all three conditions. Neither toggle touches it, and
# the study graph raises "missing segmentation for <ref>" on the FIRST bab without
# it -- so omitting this does not degrade the run, it ends it. The sample spec must
# be the one the study graph is given, or the refs written are for different babs.
echo "== segmentation (shared across A/B/C) =="
python scripts/run_segmentation.py "$COMMENTATOR" --sample "$SAMPLE" \
  --run-id ablation_seg --output-dir "$SEG_DIR"

echo "== naive readings (shared across A/B/C) =="
python scripts/run_naive_batch.py "$COMMENTATOR" --sample "$SAMPLE" --run-id "$NAIVE_RUN_ID"
python scripts/run_naive_batch.py "$COMMENTATOR" --sample "$SAMPLE" --run-id "$NAIVE_RUN_ID" --merge

declare -A LEGACY_EXAMPLE=( [A]=1 [B]=0 [C]=0 )
declare -A LEGACY_MENU=(    [A]=1 [B]=1 [C]=0 )

failed=()
for condition in A B C; do
  flags=()
  [[ "${LEGACY_EXAMPLE[$condition]}" == 1 ]] && flags+=(--legacy-example-prompt)
  [[ "${LEGACY_MENU[$condition]}" == 1 ]] && flags+=(--legacy-hypothesis-menu)

  echo "== condition $condition (legacy-example-prompt=${LEGACY_EXAMPLE[$condition]} legacy-hypothesis-menu=${LEGACY_MENU[$condition]}) =="
  # ponytail: a condition that dies must not take the other two with it. A false
  # positive in check_provenance_claim killed A under `set -e`, so B and C never
  # ran -- ~3h of unstarted work lost to a 1-line bug. Failures are collected and
  # the script still exits nonzero at the end.
  if ! (
    python scripts/run_study_graph.py "$COMMENTATOR" \
      --sample "$SAMPLE" \
      --run-id "ablation_${condition}" \
      --naive-dir "outputs/naive/${NAIVE_RUN_ID}" \
      --segmentation-dir "$SEG_DIR" \
      --notes-dir "${OUT_ROOT}/${condition}" \
      --auto-approve \
      "${flags[@]}"
  ); then
    echo "!! condition $condition FAILED -- continuing with the rest" >&2
    failed+=("$condition")
  fi
done

if (( ${#failed[@]} )); then
  echo "!! failed conditions: ${failed[*]}" >&2
  exit 1
fi

echo
echo "Done. Compare with:"
echo "  python scripts/compare_methods.py ${OUT_ROOT}/A ${OUT_ROOT}/B ${OUT_ROOT}/C"
