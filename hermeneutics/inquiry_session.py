"""One inquiry session: set up, run the graph, save. Shared by the CLI and the UI.

Extracted from scripts/run_inquiry.py so the two entry points cannot drift: a session
the UI writes and one the CLI writes are the same file with the same keys.
"""
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

from hermeneutics.conditions import CONDITIONS, load_passages
from hermeneutics.corpus_tools import CorpusTools, resolve_scope, scope_for
from hermeneutics.inquiry_graph import InquiryDeps, build_inquiry_graph, load_metrics
from hermeneutics.manifest import build_manifest, manifest_id, write_manifest
from hermeneutics.trace import TraceWriter

ROOT = Path(__file__).resolve().parent.parent

# al-Nawawi's Minhaj commentates Muslim; al-Ayni's Umdat al-Qari commentates Bukhari;
# al-Mazandarani's Sharh commentates al-Kafi. Defaulting every commentator to Bukhari
# pairs a derived method with a corpus it was never derived from -- the same class of
# defect as d042bf8 and 526f90c, both of which failed towards "the claim survives".
CORPUS_FOR = {"nawawi": "muslim", "cayni": "bukhari", "mazandarani": "kafi"}

# The study run that produced the method files. A run tag, not a constant -- there will
# be others, so --method-run overrides it.
DEFAULT_METHOD_RUN = "matched_qwen3_20260917"


def default_corpus(commentator: str) -> str:
    return CORPUS_FOR[commentator]


def method_path(commentator: str, method_run: str, root: Path = ROOT) -> Path:
    """The model's copy of the derived method file, not the human reading copy."""
    return (Path(root) / "outputs" / "method_files_matched" / method_run /
            commentator / f"{commentator}_method.md")


def load_framework(condition: str, path: Path) -> str:
    """Only agent_method receives the derived method file."""
    if CONDITIONS[condition]["needs"] != "method":
        return ""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/run_study_graph.py first "
            "to produce the method file that agent_method depends on."
        )
    return path.read_text(encoding="utf-8")


def load_passages_for(condition: str, passages_file, question: str):
    """The researcher-selected passages for THIS question, or None.

    Raising rather than degrading: a `passages` run with no passages is `no_sources`
    under a different name, and would enter the results table as a second baseline.
    """
    if CONDITIONS[condition]["needs"] != "passages":
        return None
    table = load_passages(Path(passages_file) if passages_file else None)
    found = table.get(question)
    if not found:
        raise ValueError(
            f"condition 'passages' needs researcher-selected passages for {question!r}; "
            "supply --passages-file with that question as a key."
        )
    return found


@dataclass
class RunParams:
    """What the researcher chose. Stored in ui_run.json so a resume rebuilds the run
    from these, never from whatever the form shows now."""
    question: str
    commentator: str = "nawawi"
    corpus: str | None = None          # None: the corpus the commentator commentates
    condition: str = "agent_method"
    model_id: str = ""
    max_tokens: int = 4000
    method_run: str = DEFAULT_METHOD_RUN
    passages_file: str | None = None
    auto_approve: bool = False
    run_id: str | None = None
    # Which backend served the run, and (for OpenRouter) which upstream serving
    # stack was pinned. Both are recorded so a reader of the manifest or session
    # JSON can always tell what actually answered -- "saia" is the only backend
    # that existed before this field, so it is the default an old ui_run.json
    # implicitly meant (see from_dict).
    #
    # provider_route defaults to None, not DEFAULT_PROVIDER_ROUTE: serve_ui.py builds
    # a RunParams for a SAIA run without ever setting this field, and "deepinfra" as
    # the default wrote provider_route: "deepinfra" into a SAIA run's manifest -- a
    # stack that never served it. run_inquiry.py's --provider-route flag still
    # defaults to DEFAULT_PROVIDER_ROUTE for a CLI OpenRouter run.
    provider: str = "saia"
    provider_route: str | None = None
    # The quantization pinned alongside provider_route. None for saia and for a
    # frontier OpenRouter run with no route (there is nothing to quantize-pin).
    quantization: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Prepared:
    params: RunParams
    corpus: str
    manifest: dict
    manifest_id: str
    run_id: str
    run_dir: Path
    deps: InquiryDeps
    initial_state: dict
    stamp: str
    sessions_dir: Path

    @property
    def thread_id(self) -> str:
        return f"inquiry_{self.stamp}"

    @property
    def db_path(self) -> Path:
        return self.run_dir / f"inquiry_{self.stamp}.sqlite"

    @property
    def trace_path(self) -> Path:
        return self.deps.trace.path


def new_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def prepare(params: RunParams, *, model, root: Path = ROOT, out_root: Path | None = None,
            manifest: dict | None = None, stamp: str | None = None) -> Prepared:
    """Everything before the first graph.invoke.

    model: built by the caller -- the CLI from VLLM_* env, the UI from the chosen SAIA
    model -- so this module never reads the environment itself.
    root: where metrics, method files and the corpus are read. out_root: where runs and
    sessions are written (defaults to root; tests point it at tmp_path).
    manifest / stamp: pass the stored ones on resume. A rebuilt manifest carries today's
    date and HEAD's sha, so its id would no longer match the run being resumed, and a
    new stamp would open a different checkpoint.
    """
    root = Path(root)
    out_root = Path(out_root) if out_root else root
    corpus = params.corpus or default_corpus(params.commentator)
    metrics = load_metrics(root / "outputs" / "networks" / f"{corpus}_metrics.json")
    framework = load_framework(params.condition,
                               method_path(params.commentator, params.method_run, root))
    passages = load_passages_for(params.condition, params.passages_file, params.question)

    if manifest is None:
        manifest = build_manifest(commentator=params.commentator, model_id=params.model_id)
        manifest["condition"] = params.condition
        manifest["corpus"] = corpus
        manifest["corpus_explicit"] = params.corpus is not None
        manifest["method_run"] = params.method_run
        # Part of the manifest, so it feeds manifest_id: a run whose budget was 2000 and
        # one whose budget was 4000 are not the same experiment, and the truncated calls
        # the lower budget produced are invisible in the session file.
        manifest["max_tokens"] = params.max_tokens
        # Which backend served the run, and which upstream stack OpenRouter was told
        # to pin -- a reader must always be able to tell the two apart, since a Qwen
        # run through SAIA and the same model through OpenRouter's deepinfra route
        # are not the same experiment.
        manifest["provider"] = params.provider
        manifest["provider_route"] = params.provider_route
        manifest["quantization"] = params.quantization
        # The concrete files this commentator's run may search (corpus_tools.
        # TRADITION_SCOPE), so a reader can check what the agent was allowed to see --
        # recorded even for no_sources/passages, which never touch these files, as a
        # true statement about what the run WOULD have been scoped to had it retrieved.
        manifest["scope"] = resolve_scope(root, params.commentator)
    mid = manifest_id(manifest)
    run_id = params.run_id or mid
    run_dir = out_root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, run_dir)

    deps = InquiryDeps(
        model=model,
        trace=TraceWriter(run_dir=run_dir, phase=f"inquiry_{corpus}_{params.condition}"),
        manifest_id=mid,
        metrics=metrics,
        framework=framework,
        commentator=params.commentator,
        corpus=corpus,
        # no_sources and passages must not retrieve, so they get no tools at all. The
        # rest are scoped to this commentator's own tradition (corpus_tools.
        # TRADITION_SCOPE) -- search_sources, lookup_narrator and lookup_isnad_network
        # can no longer wander into a rival corpus just because a `source`/`school`/
        # `corpus` argument named one.
        corpus_tools=CorpusTools(root, scope=scope_for(params.commentator))
                    if CONDITIONS[params.condition]["tools"] else None,
        condition=params.condition,
    )

    question = params.question
    if passages:
        body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        question = f"{params.question}\n\n--- RESEARCHER-SELECTED PASSAGES ---\n{body}"

    initial_state = {
        "condition": params.condition, "question": question,
        "commentator": params.commentator, "corpus": corpus,
        "network_metrics": metrics, "framework": framework,
        "run_manifest": manifest, "intervention_log": [],
        "sources": [], "tool_calls_log": [], "grounding_log": [],
    }
    return Prepared(params=params, corpus=corpus, manifest=manifest, manifest_id=mid,
                    run_id=run_id, run_dir=run_dir, deps=deps,
                    initial_state=initial_state, stamp=stamp or new_stamp(),
                    sessions_dir=out_root / "outputs" / "sessions")


@contextmanager
def open_graph(prepared: Prepared):
    """(graph, cfg) over this run's SQLite checkpoint. Same file and thread id on every
    call for the same Prepared, which is what makes a resume a resume."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(prepared.db_path)) as checkpointer:
        graph = build_inquiry_graph(prepared.deps, checkpointer=checkpointer)
        yield graph, {"configurable": {"thread_id": prepared.thread_id}}


def write_session(prepared: Prepared, result: dict, resumes=()) -> Path:
    """The session JSON. resumes: ISO timestamps, one per resume; empty for a run that
    went through in one piece. Kept out of the intervention log, which records human
    decisions at gates and is written only by the graph."""
    p = prepared.params
    out = prepared.sessions_dir / f"{prepared.corpus}_{p.condition}_{prepared.stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "question": p.question, "condition": p.condition, "corpus": prepared.corpus,
        # Written by the canonicalise node; empty under no_sources, which never
        # canonicalises. Recorded so a reader can see what was actually searched.
        "question_ar": result.get("question_ar", ""),
        "search_terms": result.get("search_terms", []),
        "question_lang": result.get("question_lang", ""),
        "commentator": p.commentator, "manifest_id": prepared.manifest_id,
        "method_run": p.method_run,
        # The concrete files this run could search (Part 5 item 5) -- also on the
        # manifest, repeated here so a reader of the session alone can check it.
        "scope": prepared.manifest.get("scope", []),
        "model_id": prepared.manifest.get("model_id", ""),
        "provider": prepared.manifest.get("provider", p.provider),
        "provider_route": prepared.manifest.get("provider_route", p.provider_route),
        "quantization": prepared.manifest.get("quantization", p.quantization),
        "resumes": list(resumes),
        "isnad_output": result.get("isnad_output", ""),
        "matn_output": result.get("matn_output", ""),
        "synthesis_output": result.get("synthesis_output", ""),
        "intervention_log": result.get("intervention_log", []),
        # A session must be auditable without the trace: what was read, what was called,
        # and which quotations survived the gate.
        "sources": result.get("sources", []),
        "tool_calls_log": result.get("tool_calls_log", []),
        "grounding_log": result.get("grounding_log", []),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
