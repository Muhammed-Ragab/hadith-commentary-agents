"""Complete local trace of every model call. Spec §6.3.

This is the citable research record. LangSmith is optional convenience only -
nothing in the paper may depend on it. At ~28K calls this is a few GB of JSONL.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


class TraceWriter:
    """Append-only JSONL, one line per model call, flushed immediately."""

    def __init__(self, run_dir: Path, phase: str):
        self.path = Path(run_dir) / f"{phase}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: line-buffered append handle, no rotation. Add rotation only if a
        # single phase file outgrows what the filesystem is happy with.
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def log(self, *, node: str, bab_ref: str, prompt: str, response: str,
            model_id: str, tokens_in: int, tokens_out: int, latency_s: float,
            attempt: int = 1, error: str | None = None,
            hypothesis_hash: str | None = None, manifest_id: str | None = None,
            **extra) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "node": node,
            "bab_ref": bab_ref,
            "prompt": prompt,
            "response": response,
            "model_id": model_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": round(latency_s, 3),
            "attempt": attempt,
            "error": error,
            "hypothesis_hash": hypothesis_hash,
            "manifest_id": manifest_id,
            **extra,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
