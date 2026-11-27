"""Local web UI over the Query-Agent. Spec:
docs/superpowers/specs/2026-09-22-query-agent-ui-design.md

Usage (from poc/):
  python scripts/serve_ui.py              # http://127.0.0.1:8765, models from SAIA
  python scripts/serve_ui.py --port 9000
  python scripts/serve_ui.py --fake       # canned answers, no SAIA; use no_sources;
                                           # runs/sessions are written under
                                           # outputs/ui_fake/, not beside real ones

Binds 127.0.0.1 only. The SAIA key is read from the User-scope SAIA_API_KEY and never
sent to the page. poc/.env is deliberately NOT loaded: it points at HUMAIN (403).
"""
import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from hermeneutics.conditions import CONDITIONS
from hermeneutics.inquiry_session import CORPUS_FOR, ROOT, RunParams
from hermeneutics.saia import DEFAULT_MODEL, SAIA_BASE_URL, list_models, saia_key
from scripts.build_session_view import build as build_view
from scripts.ui_runs import RunBusy, RunManager, list_sessions, read_trace

UI_DIR = Path(__file__).resolve().parent / "ui"
CORPORA = ("bukhari", "kafi", "muslim")
_STEM = re.compile(r"[\w.-]+")


class StartBody(BaseModel):
    question: str
    commentator: str = "nawawi"
    corpus: str | None = None
    condition: str = "agent_method"
    model_id: str = DEFAULT_MODEL
    max_tokens: int = 4000
    passages_file: str | None = None
    auto_approve: bool = False


class DecisionBody(BaseModel):
    action: str
    note: str | None = None


def create_app(manager: RunManager, models_fn) -> FastAPI:
    """models_fn() -> (ids, warning); injected so tests never reach SAIA."""
    app = FastAPI(title="Query-Agent")
    cache: dict = {}

    def run_or_404(run_id: str):
        try:
            return manager.get(run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/models")
    def models(refresh: int = 0):
        if refresh or "v" not in cache:
            ids, warning = models_fn()
            cache["v"] = {"models": ids,
                          "default": DEFAULT_MODEL if DEFAULT_MODEL in ids else ids[0],
                          "warning": warning}
        return cache["v"]

    @app.get("/api/options")
    def options():
        return {"commentators": list(CORPUS_FOR), "corpus_for": CORPUS_FOR,
                "corpora": list(CORPORA), "conditions": list(CONDITIONS),
                "defaults": {"commentator": "nawawi", "condition": "agent_method",
                             "max_tokens": 4000}}

    @app.post("/api/runs")
    def start(body: StartBody):
        if not body.question.strip():
            raise HTTPException(400, "the question is empty")
        if body.condition not in CONDITIONS:
            raise HTTPException(400, f"unknown condition {body.condition!r}")
        if body.commentator not in CORPUS_FOR:
            raise HTTPException(400, f"unknown commentator {body.commentator!r}")
        if body.corpus is not None and body.corpus not in CORPORA:
            raise HTTPException(400, f"unknown corpus {body.corpus!r}")
        try:
            run = manager.start(RunParams(**body.model_dump()))
        except RunBusy as e:
            raise HTTPException(409, str(e)) from None
        return run.public()

    @app.get("/api/runs/active")
    def active():
        run = manager.active()
        return {"run": run.public() if run else None}

    @app.get("/api/runs")
    def runs(status: str = ""):
        wanted = {s for s in status.split(",") if s}
        rows = sorted(manager.runs.values(), key=lambda r: r.started, reverse=True)
        return {"runs": [r.public() for r in rows if not wanted or r.status in wanted]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, trace_offset: int = 0):
        run = run_or_404(run_id)
        rows, offset = read_trace(run.trace_path, trace_offset)
        return {**run.public(), "trace": rows, "trace_offset": offset}

    def _act(fn, *args):
        try:
            return fn(*args).public()
        except KeyError:
            raise HTTPException(404, f"no run {args[0]}") from None
        except RunBusy as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/runs/{run_id}/decision")
    def decision(run_id: str, body: DecisionBody):
        return _act(manager.decide, run_id, body.action, body.note)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return _act(manager.cancel, run_id)

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str):
        return _act(manager.resume, run_id)

    @app.get("/api/sessions")
    def sessions():
        return {"sessions": list_sessions(manager.out_root)}

    @app.get("/sessions/{stem}.html", response_class=HTMLResponse)
    def session_page(stem: str):
        src = manager.out_root / "outputs" / "sessions" / f"{stem}.json"
        if not _STEM.fullmatch(stem) or ".." in stem or not src.is_file():
            raise HTTPException(404, "no such session")
        out = src.parent / "html" / f"{stem}.html"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            build_view(src, out)
        return out.read_text(encoding="utf-8")

    return app


def _fake_factory(params):
    """Canned answers for checking the page without SAIA. Tool-calling conditions need a
    model that can bind tools, which the fake cannot -- use no_sources with --fake."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    return GenericFakeChatModel(messages=iter([
        "مرحلة الإسناد: جواب تجريبي.", "مرحلة المتن: جواب تجريبي.",
        "التركيب: جواب تجريبي."]))


def build_manager(fake: bool, key: str = "") -> tuple[RunManager, Callable]:
    """manager + models_fn for main() and create_app(). Split out so it is testable
    without starting uvicorn."""
    if fake:
        # Fake sessions must never mix with real research sessions in outputs/runs or
        # outputs/sessions (e.g. the sidebar globs outputs/sessions/*.json) -- write them
        # under their own out_root instead. Reads (method files, networks) still come
        # from the real root.
        manager = RunManager(_fake_factory, out_root=ROOT / "outputs" / "ui_fake")
        return manager, lambda: ([DEFAULT_MODEL], "--fake: no SAIA calls")

    def model_factory(params: RunParams):
        from hermeneutics.llm import get_chat_model
        return get_chat_model(max_tokens=params.max_tokens, model_id=params.model_id,
                              base_url=SAIA_BASE_URL, api_key=key)

    manager = RunManager(model_factory)
    return manager, lambda: list_models(SAIA_BASE_URL, key)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Local web UI over the Query-Agent.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fake", action="store_true",
                   help="canned answers, no SAIA calls (use condition no_sources)")
    args = p.parse_args(argv)

    import uvicorn

    key = ""
    if not args.fake:
        key = saia_key()
        if not key:
            print("warning: SAIA_API_KEY is not set (User scope); runs will fail with 401",
                  file=sys.stderr)

    manager, models_fn = build_manager(args.fake, key)
    app = create_app(manager, models_fn)

    print(f"Query-Agent UI on http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
