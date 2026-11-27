"""Run manifest. Spec §3.4.

Reproducibility contract:
    same manifest + same intervention log => same outputs, up to GPU nondeterminism.
The intervention-log half lives in the notes/trace files, not here. Greedy decoding
alone is not enough: method_hypothesis accumulates into every read_commentary prompt,
so outputs are path-dependent on bāb order, scholar edits, and re-read decisions.
"""
import hashlib
import json
import subprocess
from datetime import date as _date
from pathlib import Path

PARSER_VERSION = "2"  # bump whenever scripts/parse_openiti.py changes its output

_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILES = {
    "nawawi": _ROOT / "data" / "babs" / "nawawi_babs.json",
    "cayni": _ROOT / "data" / "babs" / "cayni_babs.json",
}
METRICS_FILES = {
    "bukhari": _ROOT / "outputs" / "networks" / "bukhari_metrics.json",
    "kafi": _ROOT / "outputs" / "networks" / "kafi_metrics.json",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_manifest(commentator: str, model_id: str, weights_hash: str = "",
                   date: str | None = None) -> dict:
    from hermeneutics.prompts import prompt_hashes

    return {
        "commentator": commentator,
        "git_sha": _git_sha(),
        "model_id": model_id,
        "weights_hash": weights_hash,
        "prompt_hashes": prompt_hashes(),
        "corpus_hashes": {k: file_sha256(p) for k, p in CORPUS_FILES.items() if p.exists()},
        "metrics_hashes": {k: file_sha256(p) for k, p in METRICS_FILES.items() if p.exists()},
        "parser_version": PARSER_VERSION,
        "temperature": 0,
        "date": date or _date.today().isoformat(),
    }


def manifest_id(manifest: dict) -> str:
    """Short stable id derived from the whole manifest."""
    blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def write_manifest(manifest: dict, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return path
