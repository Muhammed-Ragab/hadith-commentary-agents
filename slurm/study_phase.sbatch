#!/bin/bash
#SBATCH --job-name=herm-study
#SBATCH --partition=grete:shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/study_%j.out

# Spec §6.2 phase 2. DELIBERATELY NOT AN ARRAY JOB.
#
# The method hypothesis evolves across units, consolidation is per-kitāb, re-reads
# depend on prior state, and interrupts suspend mid-run. Sharding the bāb list across
# array tasks cannot preserve that ordered state, and a shared SQLite checkpointer
# across concurrent writers is the wrong persistence model. Run this sequentially,
# one job per commentator; parallelism comes only from concurrent chunk requests
# inside read_commentary.
#
# Submit both commentators as two concurrent jobs to fit the overnight window:
#   sbatch --export=ALL,COMMENTATOR=nawawi slurm/study_phase.sbatch
#   sbatch --export=ALL,COMMENTATOR=cayni  slurm/study_phase.sbatch
#
# NOTE: this job needs the scholar at the terminal for 63 kitāb interrupts. For an
# unattended run, drive it interactively instead and use this script only for the
# non-interactive smoke pass (--limit).

set -euo pipefail
source "$HOME/.venvs/hermeneutics/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

COMMENTATOR="${COMMENTATOR:?set COMMENTATOR=nawawi|cayni}"
read -ra _EXTRA <<< "${STUDY_ARGS:-}"
python scripts/run_study_graph.py "$COMMENTATOR" "${_EXTRA[@]+"${_EXTRA[@]}"}"
