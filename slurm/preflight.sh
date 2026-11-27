#!/bin/bash
# Everything that can be checked WITHOUT an allocation. Run from poc/ on the login node.
#
#   bash slurm/preflight.sh
#
# Five jobs have died in startup (15518082, 15557225, 15637520, 15637631, 15637845).
# Every one of those failures was either visible in the previous job's log or checkable
# here for free. This script exists so the next allocation is spent on the pipeline
# rather than on a config error.
#
# It cannot check anything that needs a GPU: the cudagraph/mamba allocation, KV sizing,
# or generation itself. Those only fail under smoke.sbatch.

set -uo pipefail
cd "$(dirname "$0")/.."

source slurm/_vllm_common.sh
P="$(readlink -f "${PROJECT:-}")"
fail=0
note() { printf '%-8s %s\n' "$1" "$2"; [[ "$1" == "FAIL" ]] && fail=1; return 0; }

echo "=== container and vLLM ==="
module load "${APPTAINER_MOD:-apptainer/1.4.3}" 2>/dev/null || true

# The env var names this project sets. vLLM warns "Unknown vLLM environment variable"
# for a name it does not recognise and then ignores it -- silently, apart from that
# warning, which is how a wrong name would look identical to a wrong diagnosis.
apptainer exec --bind /mnt/vast-kisski:/mnt/vast-kisski --pwd "$(readlink -f "$PWD")" \
  "$VLLM_SIF" bash -lc "source '$VLLM_VENV/bin/activate' && python - <<'PY'
import vllm, vllm.envs as envs
print(f'vllm {vllm.__version__}')
for name in ('VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_USE_DEEP_GEMM'):
    print(('OK      ' if name in envs.environment_variables else 'FAIL    ') + name +
          ('' if name in envs.environment_variables else '  <- not a recognised vLLM env var'))
# Everything kernel_warmup may JIT-build. Each is a candidate for the next failure,
# since this container has the CUDA runtime but no nvcc.
import inspect
from vllm.model_executor.warmup import kernel_warmup
src = inspect.getsource(kernel_warmup)
print('kernel_warmup calls:', sorted({l.strip() for l in src.splitlines()
                                      if 'warmup(' in l and not l.strip().startswith('def')}))
PY" 2>&1 | sed 's/^/  /'

echo
echo "=== pipeline venv ==="
POC_VENV="${POC_VENV:-.venv}"
if [[ -f "$POC_VENV/bin/activate" ]]; then
  note OK "$POC_VENV exists"
  # shellcheck disable=SC1091
  source "$POC_VENV/bin/activate"
  python - <<'PY' || echo "FAIL     pipeline imports"
import importlib
for m in ("langgraph", "langchain_openai", "pydantic", "transformers"):
    try:
        importlib.import_module(m)
        print(f"OK       {m}")
    except Exception as exc:
        print(f"FAIL     {m}: {exc}")
PY
  python -c "import sys; sys.path.insert(0,'.'); import scripts.run_study_graph" \
    && note OK "run_study_graph imports" || note FAIL "run_study_graph does not import"
else
  note FAIL "$POC_VENV/bin/activate missing -- study_matched.sbatch will run against system python"
fi

echo
echo "=== corpora ==="
for c in nawawi cayni mazandarani; do
  f="data/babs/${c}_babs.json"
  [[ -s "$f" ]] && note OK "$f" || note FAIL "$f missing"
done

echo
echo "=== outputs writable ==="
d=outputs/method_files_matched/.preflight
mkdir -p "$d" 2>/dev/null && rmdir "$d" && note OK "outputs/ writable" || note FAIL "outputs/ not writable"

echo
[[ "$fail" == "0" ]] && echo "preflight clean" || echo "preflight FOUND PROBLEMS -- fix before sbatch"
exit "$fail"
