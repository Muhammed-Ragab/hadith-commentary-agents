import re
from pathlib import Path

SLURM = Path(__file__).resolve().parent.parent / "slurm"


SCRIPTS = ("batch_phase.sbatch", "study_phase.sbatch", "serve_vllm.sbatch", "bakeoff.sbatch")


def directives(name: str) -> str:
    """The script minus its prose. Comments explain what changed and why, and would
    otherwise trip every 'the old value is gone' assertion below."""
    return "\n".join(
        ln for ln in (SLURM / name).read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#") or ln.lstrip().startswith("#SBATCH")
    )


def test_all_job_scripts_exist():
    for name in SCRIPTS:
        assert (SLURM / name).exists()


def test_batch_phase_is_an_array_job():
    text = (SLURM / "batch_phase.sbatch").read_text(encoding="utf-8")
    assert "--array" in text
    assert "SLURM_ARRAY_TASK_ID" in text


def test_study_phase_is_not_an_array_job():
    """Spec §6.2: sharding the bāb list breaks the evolving hypothesis."""
    text = (SLURM / "study_phase.sbatch").read_text(encoding="utf-8")
    assert "--array" not in text


def test_study_phase_documents_why_it_is_sequential():
    text = (SLURM / "study_phase.sbatch").read_text(encoding="utf-8").lower()
    assert "hypothesis" in text or "sequential" in text


def test_batch_phase_runs_both_batch_passes():
    text = (SLURM / "batch_phase.sbatch").read_text(encoding="utf-8")
    assert "run_segmentation.py" in text
    assert "run_naive_batch.py" in text


def test_all_scripts_have_safety_header():
    for name in SCRIPTS:
        text = (SLURM / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text, f"{name}: missing set -euo pipefail"
        assert 'cd "$SLURM_SUBMIT_DIR"' in text, f"{name}: missing cd SLURM_SUBMIT_DIR"
        assert "mkdir -p logs" in text, f"{name}: missing mkdir -p logs"


COMMON = "_vllm_common.sh"


def test_serving_asks_for_a_gpu_that_fits_the_candidates():
    """Handoff §8 q2: Falcon-H1-Arabic-34B wants ~70GB bf16 plus KV cache.

    grete:shared can hand out a 40GB A100, where it does not fit.
    """
    text = directives("serve_vllm.sbatch")
    assert "grete:shared" not in text
    assert "H100" in text


def test_serving_uses_the_verified_32k_context():
    """Job 15638233 proved 32K; long commentary is reduced before the diff call."""
    text = directives("serve_vllm.sbatch")
    assert "32768" in text


def test_vllm_runs_from_the_container():
    """Serving lives in the Apptainer container (4a300c1, reverting c04b248): the host venv
    was four model-generations stale and could not import DeepseekV3Config (job 15479179)."""
    common = (SLURM / COMMON).read_text(encoding="utf-8")
    assert "apptainer exec" in common
    assert "$VLLM_SIF" in common
    assert "$VLLM_VENV/bin/activate" in common
    for name in ("serve_vllm.sbatch", "bakeoff.sbatch"):
        text = directives(name)
        assert "source slurm/_vllm_common.sh" in text, name
        assert "module load cuda" not in text, f"{name}: the venv supplies CUDA"


def test_serving_pins_decoding():
    """Spec §3.4 reproducibility: a fixed seed belongs on the server.

    The structured-outputs backend is NOT pinned here. --guided-decoding-backend was
    removed from vLLM's CLI and killed job 15504236 at argparse; the modern default
    (backend "auto") selects xgrammar anyway, and nothing client-side asks for guided
    decoding yet. Pin it via --structured-outputs-config when Phase 1 wires
    agent_schemas.py to a real call.
    """
    common = (SLURM / COMMON).read_text(encoding="utf-8")
    assert "--guided-decoding-backend" not in common
    assert "--seed 0" in common


def test_jobs_run_offline_against_the_staged_cache():
    """Models are pre-staged in $PROJECT/hf_home; a job must never try to download."""
    common = (SLURM / COMMON).read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in common
    assert "hf_home" in common


def test_missing_models_fail_before_the_allocation_is_wasted():
    """require_models runs before any server starts, so a typo fails in seconds."""
    common = (SLURM / COMMON).read_text(encoding="utf-8")
    assert "require_models()" in common
    for name in ("serve_vllm.sbatch", "bakeoff.sbatch"):
        text = directives(name)
        assert "require_models" in text, name


def test_bakeoff_uses_the_arabic_falcon_not_the_general_one():
    """Both are in the cache. The Arabic line is the one whose hypothesis this tests."""
    text = directives("bakeoff.sbatch")
    assert "Falcon-H1-Arabic-34B-Instruct" in text
    assert "tiiuae/Falcon-H1-34B-Instruct" not in text


def test_bakeoff_candidates_span_the_distinct_hypotheses():
    """Arabic-native / general-frontier / cheap-floor. Drop one and it stops being a
    bake-off and becomes a beauty contest (handoff §8.1)."""
    text = directives("bakeoff.sbatch")
    for model in ("Falcon-H1-Arabic-34B-Instruct", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"):
        assert model in text, model


def test_bakeoff_includes_the_incumbent():
    """gemma-4-31B-it produced the existing study notes. Without it in the run there is no
    counterfactual for switching, and the paper cannot claim the switch earned anything."""
    text = directives("bakeoff.sbatch")
    assert "gemma-4-31B-it" in text


def test_bakeoff_has_a_gpu_per_candidate():
    """Candidates are served concurrently, one per card."""
    text = directives("bakeoff.sbatch")
    block = re.search(r"CANDIDATES=\((.*?)\)", text, re.S).group(1)
    n = len([ln for ln in block.splitlines() if ln.strip().startswith('"')])
    gres = re.search(r"--gres=gpu:H100:(\d+)", text).group(1)
    assert int(gres) >= n, f"{n} candidates but only {gres} GPUs requested"


def test_models_needing_download_have_a_staging_path():
    """Compute nodes are offline, so anything not cached must be fetched on the login node."""
    stage = (SLURM / "stage_models.sh").read_text(encoding="utf-8")
    assert "Qwen/Qwen3.8-27B" in stage
    assert "Qwen/Qwen3.6-35B-A3B" in stage
    assert "stage_models.sh" in (SLURM / "bakeoff.sbatch").read_text(encoding="utf-8")


def test_bakeoff_isolates_candidates():
    """Notes and traces are keyed per candidate, or they overwrite each other."""
    text = (SLURM / "bakeoff.sbatch").read_text(encoding="utf-8")
    assert "--notes-dir" in text
    assert "--run-id" in text
    assert "--auto-approve" in text, "batch jobs cannot answer input() at a kitāb interrupt"
    assert "score_bakeoff.py" in text


def test_matched_job_isolates_runs_segments_inputs_and_validates_outputs():
    text = (SLURM / "study_matched.sbatch").read_text(encoding="utf-8")
    assert "SLURM_JOB_ID" in text
    assert "run_segmentation.py" in text and '--limit "$N_BABS"' in text
    assert "--segmentation-dir" in text
    assert "--live-naive" in text
    assert "validate_matched.py" in text


def test_matched_job_supports_single_or_multi_gpu_serving():
    text = directives("study_matched.sbatch")
    assert '"${VLLM_TP:-1}"' in text


def test_matched_job_defaults_qwen_to_the_reasoning_parser():
    """Default is unchanged; VLLM_REASONING_PARSER=none is the escape hatch.

    The parser is the suspect for Qwen's measured 7,697 s/bab, so it has to be
    switchable without editing the file -- but a plain run must still behave as before.
    """
    text = directives("study_matched.sbatch")
    assert 'RP="${VLLM_REASONING_PARSER:-auto}"' in text
    assert '[[ "$RP" == auto && "$STUDY_MODEL" == Qwen/Qwen3.* ]] && RP=qwen3' in text
    assert '[[ "$RP" != auto && "$RP" != none ]] && server_args+=(--reasoning-parser "$RP")' in text


def test_matched_job_can_swap_the_vllm_launcher():
    """Qwen3.8-27B needs the host-venv stack; Falcon keeps the container one."""
    text = directives("study_matched.sbatch")
    assert 'source "slurm/${VLLM_LAUNCHER:-_vllm_common.sh}"' in text


def test_host_venv_launcher_reuses_the_shared_helpers():
    text = directives("_vllm_venv.sh")
    assert "_vllm_common.sh" in text          # resolve_model / require_models / wait_for_server
    assert "serve_model()" in text
    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in text
    assert "apptainer" not in text


def test_login_preflight_checks_the_small_arabic_test_model():
    from slurm.check_model_configs import CANDIDATES

    assert "ALLaM-7B-Instruct-preview" in CANDIDATES


def test_matched_job_runs_arms_concurrently():
    """Serial arms left vLLM at concurrency 1.00x and made the job their sum."""
    text = directives("study_matched.sbatch")
    assert "arm_pids+=($!)" in text
    assert 'wait "${arm_pids[$i]}"' in text
    assert 'run_arm "$c" > "logs/matched_${SLURM_JOB_ID}_${c}.log" 2>&1 &' in text


def test_matched_job_run_tag_is_overridable_for_resume():
    """A tag containing the job id gives every job a fresh checkpointer."""
    text = directives("study_matched.sbatch")
    assert 'run_tag="${RUN_TAG:-matched_${slug}_${SLURM_JOB_ID}}"' in text


def test_matched_job_batches_the_naive_pass_by_default():
    text = directives("study_matched.sbatch")
    assert '"${STUDY_NAIVE:-batch}" == live' in text
    assert "run_naive_batch.py" in text
    assert "--merge" in text
