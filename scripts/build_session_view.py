"""Render one inquiry session (`outputs/sessions/*.json`) as a standalone HTML page.

Two audiences, one page:

  - The talk. `tool_calls_log` is the only evidence that this is an agent rather than
    three chained prompts, and it is invisible in every artifact we currently produce.
    The trail is rendered first, by stage, so a screenshot carries the agentic claim.
  - The Hossam Ouf sitting. `build_review_sheets.py` renders method files; nothing
    renders an inquiry session, so a scholar cannot review Query-Agent output at all.
    The quotations panel shows each quotation with its verified/not verdict, the
    `found_in` source it was actually located in, and the stripped ones struck through
    in place, so the reviewer sees what the agent claimed AND what the gate removed.

The gate rewrites `synthesis_output` before it is stored: a stripped quotation leaves
the scar "[unverified quotation removed]" and the text itself is gone from the file.
Only `grounding_log` still holds it, so the scars are paired back to their log entries
here -- that restoration is what makes the removal legible rather than a blank.

No dependencies: stdlib only, one self-contained file, no JS. Output opens from disk.

Usage:
  python scripts/build_session_view.py outputs/sessions/<session>.json
  python scripts/build_session_view.py outputs/sessions/<session>.json -o page.html
  python scripts/build_session_view.py outputs/sessions/           # whole directory
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCAR = "[unverified quotation removed]"

STAGES = ("isnad", "matn", "synthesis")
STAGE_TITLE = {"isnad": "Isnād stage", "matn": "Matn stage", "synthesis": "Synthesis"}

# Flags the gate RECORDS but never strips on (inquiry_graph._synthesis_gate). They are
# the reviewer's business, not the gate's, so they surface as badges rather than
# changing the verdict.
ADVISORY = {
    "foreign_commentary": ("rival commentary",
                           "verified, but found only in another commentator's book"),
    # A superset of foreign_commentary: it also catches a verified quotation found only
    # in a RIVAL HADITH COLLECTION, which foreign_commentary deliberately ignores
    # because quoting the hadith corpus itself is legitimate. Absent on a session run
    # before Part 5 (no `scope` recorded) -- see reverify(), which leaves it False
    # for one, exactly like a missing `commentator` would.
    "outside_scope": ("outside this run's scope",
                      "verified, but found only outside the files this run was scoped to"),
    "echoes_question": ("echoes the question",
                        "the 'quotation' is a fragment of the question itself"),
    "elided": ("elided", "matched with an ellipsis, as a scholar's citation does"),
    "in_session": ("in session", "occurs in a source this session actually retrieved"),
}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def is_arabic(s: str) -> bool:
    """Enough Arabic to warrant RTL. A reference line mixes scripts; the majority wins."""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return arabic > len(letters) / 2


def dir_attr(s: str) -> str:
    return ' dir="rtl" class="rtl"' if is_arabic(s) else ' dir="ltr"'


def clip(value, limit: int = 70) -> str:
    """`verify_quote` is called with the whole quotation -- 263 characters in one
    observed session. Rendered in full it buries the trail it is meant to demonstrate,
    which is the one thing a slide needs to read at a glance. Full text stays in the
    element's title.
    """
    s = str(value)
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def md(text: str) -> str:
    """The stage outputs are model markdown in a narrow dialect: #/##/### headings,
    > blockquotes, - bullets, **bold**. A markdown library is not installed and this
    does not warrant adding one.

    ponytail: handles only what the model actually emits; reach for a real parser if a
    prompt ever asks for tables or nested lists.
    """
    out, in_list = [], False
    for raw in esc(text).split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{min(level + 2, 6)}>{inline(stripped.lstrip('# '))}"
                       f"</h{min(level + 2, 6)}>")
        elif stripped.startswith("&gt;"):
            out.append(f"<blockquote>{inline(stripped[4:].strip())}</blockquote>")
        else:
            out.append(f"<p>{inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def restore_scars(text: str, log: list) -> str:
    """Put each stripped quotation back into the synthesis, struck through.

    The gate strips in the order `quotations()` yields, which is document order, so the
    Nth scar in the text is the Nth unverified entry in the log.

    ponytail: pairs positionally. `_strip_quote`'s regex-miss fallback uses str.replace,
    which would collapse two identical fabrications into one pairing -- rare enough to
    accept, and it degrades to an unrestored scar rather than a wrong attribution.
    """
    # Anchored to the verdict that WROTE the scars, which reverify() preserves. Using
    # the current verdict would mis-pair every session whose gate has since been fixed.
    removed = [e for e in log
               if (e["stripped_at_run_time"] if "stripped_at_run_time" in e
                   else not e.get("verified"))]
    parts = text.split(SCAR)
    if len(parts) - 1 != len(removed):
        # Trust nothing: leave the scars as the gate wrote them.
        return text
    out = parts[0]
    for entry, tail in zip(removed, parts[1:]):
        quote = esc(entry.get("quote", ""))
        if entry.get("verified"):
            # Stripped by the gate that ran, verified by the gate as it stands. The
            # model was right; show the words, and say why they are back.
            found = ", ".join(entry.get("found_in") or []) or "the corpus"
            out += (f'<span class="rescued" title="{esc(found)}">{quote}'
                    f'<span class="why-ok"> — stripped at run time, '
                    f'verifies now against {esc(found)}</span></span>') + tail
            continue
        reasons = [label for key, (label, _) in ADVISORY.items()
                   if key != "in_session" and entry.get(key)]
        why = "; ".join(reasons) or "not found in the corpus"
        out += (f'<span class="stripped" title="{esc(why)}">'
                f'<s>{quote}</s>'
                f'<span class="why"> — removed: {esc(why)}</span></span>') + tail
    return out


def tool_trail(calls: list) -> str:
    if not calls:
        return '<p class="empty">No tool calls recorded — this condition ran without tools.</p>'
    rows = []
    for stage in STAGES:
        staged = [c for c in calls if c.get("stage") == stage]
        if not staged:
            continue
        rows.append(f'<h3>{esc(STAGE_TITLE.get(stage, stage))}'
                    f' <span class="count">{len(staged)} calls</span></h3><ol class="trail">')
        for call in staged:
            args = call.get("args", {}) or {}
            argbits = "".join(
                f'<span class="arg"><span class="k">{esc(k)}</span>'
                f'<span class="v"{dir_attr(str(v))} title="{esc(v)}">{esc(clip(v))}</span>'
                "</span>"
                for k, v in args.items())
            rows.append(f'<li><code class="tool">{esc(call.get("tool"))}</code>{argbits}</li>')
        rows.append("</ol>")
    # A stage the loop never labelled would vanish silently; show it rather than lose it.
    other = [c for c in calls if c.get("stage") not in STAGES]
    if other:
        rows.append(f'<h3>Unlabelled <span class="count">{len(other)} calls</span></h3>'
                    "<ol class='trail'>" +
                    "".join(f'<li><code class="tool">{esc(c.get("tool"))}</code></li>'
                            for c in other) + "</ol>")
    return "\n".join(rows)


def quotations_panel(log: list) -> str:
    if not log:
        return '<p class="empty">No quotations were checked in this session.</p>'
    cards = []
    for entry in log:
        ok = bool(entry.get("verified"))
        badges = []
        for key, (label, tip) in ADVISORY.items():
            if entry.get(key):
                cls = ("badge warn" if key in ("foreign_commentary", "outside_scope",
                                               "echoes_question") else "badge")
                badges.append(f'<span class="{cls}" title="{esc(tip)}">{esc(label)}</span>')
        found = entry.get("found_in") or []
        where = ("".join(f'<li>{esc(s)}</li>' for s in found) if found
                 else '<li class="empty">not found in any corpus file</li>')
        quote = esc(entry.get("quote", ""))
        cards.append(f"""
        <article class="quote {'ok' if ok else 'bad'}">
          <header><span class="verdict">{'✓ verified' if ok else '✗ stripped'}</span>
            <span class="checked">{esc(entry.get('checked', 0))} source(s) searched</span>
            {''.join(badges)}</header>
          <blockquote{dir_attr(entry.get('quote', ''))}>{quote}</blockquote>
          <div class="found"><span class="label">found in</span><ul>{where}</ul></div>
        </article>""")
    verified = sum(1 for e in log if e.get("verified"))
    rescued = sum(1 for e in log if e.get("stripped_at_run_time") and e.get("verified"))
    head = (f'<p class="tally"><strong>{verified}</strong> of <strong>{len(log)}</strong> '
            "quotations verify against the corpus. The rest were removed from the "
            "synthesis before this page was built.")
    if rescued:
        # Saying which gate produced which number, rather than quietly reporting the
        # new one against a synthesis the old one wrote.
        head += (f' <strong>{rescued}</strong> of them were stripped by the gate as it '
                 "ran and verify against the corpus as it stands — this page was built "
                 "with <code>--reverify</code>, so those are shown restored.")
    return head + "</p>" + "\n".join(cards)


def sources_block(sources: list) -> str:
    if not sources:
        return ""
    items = "".join(
        f'<li><span class="src">{esc(s.get("source"))}</span>'
        f'<span class="ref"{dir_attr(str(s.get("reference", "")))}>'
        f'{esc(s.get("reference"))}</span></li>'
        for s in sources)
    return (f'<details><summary>{len(sources)} sources retrieved</summary>'
            f'<ul class="sources">{items}</ul></details>')


def searched_as_block(session: dict) -> str:
    """The canonical Arabic question and search terms, next to the original --
    absent on a session from before Part 2 (no_sources sessions never had them
    either), so this renders nothing rather than an empty, misleading panel."""
    question_ar = session.get("question_ar", "")
    terms = session.get("search_terms", []) or []
    if not question_ar:
        return ""
    terms_html = "".join(f'<span class="tool">{esc(t)}</span> ' for t in terms)
    return (f'<div class="question"{dir_attr(question_ar)}>'
            f'<div class="cell"><span class="k">Searched as</span></div>'
            f'{esc(question_ar)}'
            f'{f"<div>{terms_html}</div>" if terms_html else ""}'
            f'</div>')


def render(session: dict) -> str:
    question = session.get("question", "")
    log = session.get("grounding_log", []) or []
    meta = [("condition", session.get("condition")),
            ("commentator", session.get("commentator")),
            ("corpus", session.get("corpus")),
            ("method run", session.get("method_run")),
            ("manifest", session.get("manifest_id"))]
    meta_html = "".join(f'<span class="cell"><span class="k">{esc(k)}</span>'
                        f'<span class="v">{esc(v) or "—"}</span></span>'
                        for k, v in meta)
    # Editorial corrections to the model's text are shown, never applied silently: a
    # page that is screenshot for a talk and read by a scholar must not present edited
    # prose as the model's own. The raw trace keeps the original.
    fixes = session.get("corrections") or []
    if fixes:
        items = "".join(f'<li><span dir="rtl">{esc(c.get("from"))} → {esc(c.get("to"))}</span>'
                        f' ×{esc(c.get("count"))} in {esc(c.get("field"))}</li>'
                        for c in fixes)
        meta_html += (f'<div class="corrections"><strong>Editorial corrections to the '
                      f'model\'s text</strong><ul>{items}</ul></div>')

    stages = []
    for stage in STAGES:
        body = session.get(f"{stage}_output", "") or ""
        if not body.strip():
            continue
        rendered = md(body)
        if stage == "synthesis":
            rendered = restore_scars(rendered, log)
        stages.append(f'<section class="stage"><h2>{esc(STAGE_TITLE[stage])}</h2>'
                      f'<div class="body"{dir_attr(body)}>{rendered}</div></section>')

    return f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Query-Agent session — {esc(session.get('commentator'))}/{esc(session.get('condition'))}</title>
<style>
  :root {{ --ink:#1a1a1a; --dim:#666; --line:#ddd; --bg:#fff; --panel:#f7f7f5;
           --ok:#1a7f43; --bad:#b3261e; --warn:#8a6100; --accent:#2c4c7c; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); line-height:1.6;
          font-family:"Charter","Georgia",serif; }}
  .wrap {{ max-width:56rem; margin:0 auto; padding:2rem 1.5rem 5rem; }}
  .rtl {{ font-family:"Amiri","Scheherazade New","Traditional Arabic",serif;
          font-size:1.12em; line-height:2; text-align:right; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem;
        border-bottom:2px solid var(--ink); }}
  h3 {{ font-size:.95rem; margin:1.25rem 0 .4rem; color:var(--accent); }}
  .meta {{ display:flex; flex-wrap:wrap; gap:.5rem 1.5rem; padding:.75rem 0 1rem;
           border-bottom:1px solid var(--line); font-size:.8rem;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .cell .k {{ color:var(--dim); margin-right:.4rem; }}
  .cell .v {{ font-weight:600; }}
  .question {{ background:var(--panel); border-right:4px solid var(--accent);
               padding:1rem 1.25rem; margin:1.25rem 0; }}
  .trail {{ margin:0; padding-right:1.25rem; padding-left:1.25rem; }}
  .trail li {{ margin:.3rem 0; }}
  .tool {{ background:#eef2f8; color:var(--accent); padding:.1rem .4rem;
           border-radius:3px; font-size:.85rem; font-weight:600; }}
  .arg {{ display:inline-block; margin-inline-start:.6rem; font-size:.85rem; }}
  .arg .k {{ color:var(--dim); }}
  .arg .k:after {{ content:"="; }}
  .arg .v {{ color:var(--ink); }}
  .count {{ color:var(--dim); font-weight:400; font-size:.8rem; }}
  .tally {{ background:var(--panel); padding:.6rem .9rem; font-size:.9rem; }}
  .quote {{ border:1px solid var(--line); border-left:4px solid var(--line);
            padding:.75rem 1rem; margin:.75rem 0; }}
  .quote.ok {{ border-left-color:var(--ok); }}
  .quote.bad {{ border-left-color:var(--bad); background:#fdf6f6; }}
  .quote header {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center;
                   font-size:.78rem; margin-bottom:.4rem; }}
  .verdict {{ font-weight:700; }}
  .ok .verdict {{ color:var(--ok); }}
  .bad .verdict {{ color:var(--bad); }}
  .checked {{ color:var(--dim); }}
  .badge {{ border:1px solid var(--line); border-radius:99px; padding:.05rem .5rem;
            color:var(--dim); }}
  .badge.warn {{ border-color:var(--warn); color:var(--warn); }}
  .quote blockquote {{ margin:.25rem 0; padding:0; }}
  .found {{ font-size:.78rem; color:var(--dim); margin-top:.4rem; }}
  .found ul {{ margin:.1rem 0; padding-inline-start:1.25rem; }}
  .found .label {{ text-transform:uppercase; letter-spacing:.06em; }}
  .stage .body blockquote {{ border-inline-start:3px solid var(--line);
                             margin-inline-start:0; padding-inline-start:1rem;
                             color:#333; }}
  .stripped {{ background:#fdeceb; }}
  .stripped s {{ color:var(--bad); }}
  .why {{ font-size:.75rem; color:var(--bad); white-space:nowrap; }}
  .corrections {{ flex-basis:100%; font-family:"Charter","Georgia",serif;
                  font-size:.8rem; color:var(--warn); }}
  .corrections ul {{ margin:.2rem 0; padding-inline-start:1.25rem; }}
  .rescued {{ background:#eaf6ee; }}
  .why-ok {{ font-size:.75rem; color:var(--ok); }}
  .sources {{ font-size:.8rem; padding-inline-start:1.25rem; }}
  .sources .src {{ font-family:ui-monospace,monospace; color:var(--accent);
                   margin-inline-end:.5rem; }}
  details summary {{ cursor:pointer; margin-top:2rem; color:var(--dim);
                     font-size:.85rem; }}
  .empty {{ color:var(--dim); font-style:italic; }}
  code {{ font-size:.9em; }}
  @media print {{ .wrap {{ max-width:none; }} details {{ display:none; }} }}
</style>
<div class="wrap">
  <h1>Query-Agent session</h1>
  <div class="meta">{meta_html}</div>
  <div class="question"{dir_attr(question)}>{esc(question)}</div>
  {searched_as_block(session)}

  <h2>What the agent did</h2>
  {tool_trail(session.get("tool_calls_log", []) or [])}

  <h2>Quotations, checked against the corpus</h2>
  {quotations_panel(log)}

  {"".join(stages)}

  {sources_block(session.get("sources", []) or [])}
</div>
</html>
"""


def reverify(session: dict) -> int:
    """Re-run the gate over the quotations this session recorded.

    grounding_log is frozen at run time, so a session run before a gate fix carries that
    gate's verdicts forever. Two such fixes have landed (the `~~` artifact, then
    punctuation and bab headings), and between them they rescue 8 of the 26 strips in
    the sessions recorded on 2026-09-21. Rendered from the stored log, those 8 appear
    struck through as fabrications -- in a talk, and in front of the reviewing scholar,
    that is the tool accusing the model of inventing al-Nawawi's actual words.

    This re-verifies rather than re-runs: the model's output is untouched, only the
    verdict on it. Returns how many verdicts changed, so the caller can say so.
    """
    # Imported here, not at module scope: rendering a page needs no corpus, and the
    # tests exercise the pairing logic without loading 122k records.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hermeneutics.conditions import verify_quotation
    from hermeneutics.corpus_tools import CorpusTools, outside_scope

    # Whole-corpus on purpose (no `scope=`): verification stays unscoped so outside_scope
    # below can even be computed. A session without a `commentator` (should not happen,
    # but an old file is data, not a guarantee) leaves outside_scope False, same as
    # corpus_tools.outside_scope already does for an unconfigured commentator.
    corpus = CorpusTools(ROOT)
    commentator = session.get("commentator", "")
    changed = 0
    for entry in session.get("grounding_log", []) or []:
        # The synthesis text carries one scar per quotation the ORIGINAL gate stripped,
        # so the pairing in restore_scars must stay anchored to that verdict however the
        # new one comes out. Recording it is what lets the page show a rescued quotation
        # in place rather than as a blank the counts no longer explain.
        entry["stripped_at_run_time"] = not entry.get("verified")
        outcome = verify_quotation(entry.get("quote", ""), corpus)
        if bool(outcome["verified"]) != bool(entry.get("verified")):
            changed += 1
        entry["verified"] = bool(outcome["verified"])
        entry["elided"] = outcome["elided"]
        entry["checked"] = outcome["checked"]
        entry["found_in"] = outcome.get("sources", [])
        entry["outside_scope"] = entry["verified"] and outside_scope(
            entry["found_in"], commentator, ROOT)
    return changed


def build(path: Path, out: Path, recheck: bool = False) -> tuple:
    session = json.loads(path.read_text(encoding="utf-8"))
    changed = reverify(session) if recheck else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(session), encoding="utf-8")
    return out, changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path, help="a session JSON, or a directory of them")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output file (single session) or directory")
    p.add_argument("--reverify", action="store_true",
                   help="re-check the recorded quotations against the corpus as it "
                        "stands, instead of trusting the verdicts the gate wrote at "
                        "run time. Use this for any session recorded before a gate fix.")
    args = p.parse_args(argv)

    paths = (sorted(args.session.glob("*.json")) if args.session.is_dir()
             else [args.session])
    if not paths:
        print(f"no session JSON under {args.session}", file=sys.stderr)
        return 1
    outdir = args.out if (args.out and (args.session.is_dir() or args.out.is_dir())) else None
    for path in paths:
        out = (outdir / f"{path.stem}.html") if outdir else (
            args.out if args.out and len(paths) == 1 else path.with_suffix(".html"))
        written, changed = build(path, out, recheck=args.reverify)
        note = f"  ({changed} verdicts changed)" if changed else ""
        print(f"{written}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
