"""Background inquiry runs for the local UI. Spec:
docs/superpowers/specs/2026-09-22-query-agent-ui-design.md

No web framework here: serve_ui.py maps HTTP onto RunManager, and the tests drive it
directly. One run at a time -- SAIA throttles, and a 429 costs the whole run.
"""
import json
import queue
import re
import threading
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langgraph.types import Command

from hermeneutics.inquiry_session import (
    ROOT,
    RunParams,
    new_stamp,
    open_graph,
    prepare,
    write_session,
)

ACTIVE = {"running", "awaiting_review"}
RESUMABLE = {"failed", "interrupted"}
_CANCEL = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunBusy(RuntimeError):
    pass


@dataclass
class Run:
    run_id: str
    params: RunParams
    stamp: str
    run_dir: Path
    manifest: dict | None = None
    status: str = "running"
    stage: str | None = None
    output: str | None = None
    error: str | None = None
    session_path: str | None = None
    resumes: list = field(default_factory=list)
    started: str = field(default_factory=_now)
    trace_path: str | None = None
    cancel_requested: bool = False
    decisions: queue.Queue = field(default_factory=queue.Queue, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {
            "run_id": self.run_id, "status": self.status, "stage": self.stage,
            "output": self.output if self.status == "awaiting_review" else None,
            "error": self.error,
            "session_stem": Path(self.session_path).stem if self.session_path else None,
            "resumes": list(self.resumes), "started": self.started,
            "cancel_requested": self.cancel_requested, "params": self.params.to_dict(),
        }

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "ui_run.json").write_text(json.dumps({
            "run_id": self.run_id, "params": self.params.to_dict(), "stamp": self.stamp,
            "manifest": self.manifest, "status": self.status, "stage": self.stage,
            "error": self.error, "session_path": self.session_path,
            "resumes": self.resumes, "started": self.started,
            "trace_path": self.trace_path,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Run":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(run_id=d["run_id"], params=RunParams.from_dict(d["params"]),
                   stamp=d["stamp"], run_dir=path.parent, manifest=d.get("manifest"),
                   status=d.get("status", "interrupted"), stage=d.get("stage"),
                   error=d.get("error"), session_path=d.get("session_path"),
                   resumes=d.get("resumes", []), started=d.get("started") or _now(),
                   trace_path=d.get("trace_path"))


class RunManager:
    def __init__(self, model_factory, *, root: Path = ROOT, out_root: Path | None = None):
        """model_factory(params) -> chat model. The server builds a SAIA model from
        params.model_id; tests pass fakes."""
        self.model_factory = model_factory
        self.root = Path(root)
        self.out_root = Path(out_root) if out_root else self.root
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._load_leftovers()

    def _runs_dir(self) -> Path:
        return self.out_root / "outputs" / "runs"

    def _load_leftovers(self) -> None:
        """A run the previous server left active has no thread any more. Its checkpoint
        survives, so it is listed as interrupted and can be resumed."""
        for f in self._runs_dir().glob("*/ui_run.json"):
            try:
                run = Run.load(f)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if run.status in ACTIVE:
                run.status = "interrupted"
                run.save()
            self.runs[run.run_id] = run

    def get(self, run_id: str) -> Run:
        return self.runs[run_id]          # KeyError -> 404 in the server

    def active(self) -> Run | None:
        return next((r for r in self.runs.values() if r.status in ACTIVE), None)

    def start(self, params: RunParams) -> Run:
        with self._lock:
            if self.active():
                raise RunBusy("a run is already active; one at a time, SAIA throttles")
            stamp = new_stamp()
            # Not the manifest id: the CLI's default reuses one run dir (and one trace
            # file) for every run of the same manifest, which would mix trails.
            params.run_id = params.run_id or f"ui_{stamp}"
            run = Run(run_id=params.run_id, params=params, stamp=stamp,
                      run_dir=self._runs_dir() / params.run_id)
            run.save()
            self.runs[run.run_id] = run
        self._spawn(run, resume=False)
        return run

    def resume(self, run_id: str) -> Run:
        with self._lock:
            run = self.get(run_id)
            if run.status not in RESUMABLE:
                raise ValueError(f"run is {run.status}; only failed or interrupted runs resume")
            if self.active():
                raise RunBusy("a run is already active; one at a time, SAIA throttles")
            run.status, run.error, run.cancel_requested = "running", None, False
            run.decisions = queue.Queue()
            run.resumes.append(_now())
            run.save()
        self._spawn(run, resume=True)
        return run

    def decide(self, run_id: str, action: str, note: str | None = None) -> Run:
        run = self.get(run_id)
        if action not in ("approve", "edit"):
            raise ValueError(f"unknown action {action!r}")
        with self._lock:
            if run.status != "awaiting_review":
                raise ValueError(f"run is {run.status}, not waiting for a decision")
            decision = {"action": action}
            if action == "edit":
                decision["note"] = note or ""
            # Set here, not only in the worker, so a second click or a poll that lands
            # before the worker wakes cannot see the old gate still open.
            run.status = "running"
            run.decisions.put(decision)
        return run

    def cancel(self, run_id: str) -> Run:
        run = self.get(run_id)
        with self._lock:
            if run.status in RESUMABLE:          # dismiss a stopped run
                self._set(run, "cancelled")
                return run
            if run.status not in ACTIVE:
                raise ValueError(f"run is {run.status}; nothing to cancel")
            # An in-flight SAIA call cannot be aborted; the worker checks this flag at
            # the next gate. The page says so rather than pretending the run stopped.
            run.cancel_requested = True
            if run.status == "awaiting_review":
                run.decisions.put(_CANCEL)
        return run

    def _spawn(self, run: Run, *, resume: bool) -> None:
        run.thread = threading.Thread(target=self._drive, args=(run, resume), daemon=True)
        run.thread.start()

    def _set(self, run: Run, status: str) -> None:
        run.status = status
        run.save()

    def _drive(self, run: Run, resume: bool) -> None:
        prepared = None
        try:
            prepared = prepare(run.params, model=self.model_factory(run.params),
                               root=self.root, out_root=self.out_root,
                               manifest=run.manifest, stamp=run.stamp)
            run.manifest = prepared.manifest
            run.trace_path = str(prepared.trace_path)
            run.save()
            with open_graph(prepared) as (graph, cfg):
                # A run that failed before its first checkpoint has nothing to resume
                # from; start it from the top instead of invoking an empty thread. On a
                # fresh SqliteSaver thread, get_state(cfg) is a real (empty) StateSnapshot
                # whose .values is {} -- never None -- so this bool() check is safe.
                has_state = bool(graph.get_state(cfg).values)
                result = graph.invoke(prepared.initial_state if not (resume and has_state)
                                      else None, cfg)
                while "__interrupt__" in result:
                    payload = result["__interrupt__"][0].value
                    run.stage = payload.get("stage")
                    run.output = payload.get("output", "")
                    if run.cancel_requested:
                        return self._set(run, "cancelled")
                    if run.params.auto_approve:
                        decision = {"action": "auto_approve"}
                    else:
                        self._set(run, "awaiting_review")
                        decision = run.decisions.get()
                        if decision is _CANCEL:
                            return self._set(run, "cancelled")
                    self._set(run, "running")
                    result = graph.invoke(Command(resume=decision), cfg)
                if run.cancel_requested:
                    return self._set(run, "cancelled")
                run.session_path = str(write_session(prepared, result, resumes=run.resumes))
                self._set(run, "done")
        except Exception as e:  # noqa: BLE001 - a run failure must become status
            # "failed", not kill this daemon thread silently and leave the UI waiting
            # forever on a run that will never update again.
            tail = "".join(traceback.format_exc().splitlines(True)[-12:])
            run.error = f"{type(e).__name__}: {e}\n\n{tail}"
            self._set(run, "failed")
        finally:
            if prepared is not None:
                prepared.deps.trace.close()


def summarise_trace(rec: dict) -> dict:
    """One trace record -> one row of the live trail."""
    tools = []
    if rec.get("tool_calls"):
        try:
            msg = json.loads(rec.get("response") or "{}")
            tools = [{"name": c["function"]["name"],
                      "args": (c["function"].get("arguments") or "")[:200]}
                     for c in msg.get("tool_calls", [])]
        except (ValueError, KeyError, TypeError):
            tools = [{"name": n, "args": ""} for n in rec["tool_calls"]]
    return {
        "ts": rec.get("ts"), "node": rec.get("node"),
        "tokens_in": rec.get("tokens_in", 0), "tokens_out": rec.get("tokens_out", 0),
        "latency_s": rec.get("latency_s", 0), "error": rec.get("error"),
        "tools": tools,
        "preview": "" if tools else (rec.get("response") or "")[:160],
    }


def read_trace(path, offset: int = 0) -> tuple[list[dict], int]:
    """Rows appended after byte `offset`, and the new offset. A line still being written
    (no newline yet) is left for the next poll."""
    if not path or not Path(path).exists():
        return [], offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    rows = [summarise_trace(json.loads(line))
            for line in chunk[:end].decode("utf-8").splitlines() if line.strip()]
    return rows, offset + end + 1


_STAMP = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})")


def list_sessions(out_root: Path) -> list[dict]:
    d = Path(out_root) / "outputs" / "sessions"
    rows = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = _STAMP.search(p.stem)
        rows.append({
            "stem": p.stem, "question": (s.get("question") or "")[:160],
            "commentator": s.get("commentator", ""), "condition": s.get("condition", ""),
            "corpus": s.get("corpus", ""), "model_id": s.get("model_id", ""),
            "date": f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else "",
        })
    return rows
