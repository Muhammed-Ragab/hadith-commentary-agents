"""Render the companion PDF's single HTML page: every final-batch answer, verbatim.

Reuses the session viewer's markdown, escaping and scar restoration so the PDF and the
slide screenshots show verification the same way. Answer text is never altered: marks
are spans around exact substrings, and every added annotation carries class
ann/why/why-ok so tests can strip it and compare the words to the stored text.
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_session_view import (inline, ADVISORY, STAGES, STAGE_TITLE, dir_attr, esc, md,
                                        restore_scars)
from scripts import companion_data as cd


def mark_quotes(html_text: str, log: list, agent: bool) -> str:
    out, cursor = html_text, 0
    for n, entry in enumerate(log, 1):
        ok = bool(entry.get("verified"))
        if agent and not ok:
            continue  # a scar; restore_scars handles it
        needle = esc(entry.get("quote", ""))
        if not needle:
            continue
        at = out.find(needle, cursor)
        if at < 0:
            continue
        cls, sign = ("ok", "✓") if ok else ("no", "✗")
        wrapped = (f'<span class="q {cls}" data-n="{n}">{needle}</span>'
                   f'<sup class="ann">[{n}{sign}]</sup>')
        out = out[:at] + wrapped + out[at + len(needle):]
        cursor = at + len(wrapped)
    return out


def _cells(line: str) -> list:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def md_with_tables(text: str) -> str:
    """md() has no tables. Runs of lines starting with | become an HTML table (the
    |---| separator row is layout, not content); everything else goes through md()."""
    out, block, rows = [], [], []

    def flush_block():
        if block:
            out.append(md("\n".join(block)))
            block.clear()

    def flush_rows():
        if rows:
            body = "".join(
                "<tr>" + "\n".join(f"<td{dir_attr(c)}>{inline(esc(c))}</td>" for c in r)
                + "</tr>\n" for r in rows)
            out.append(f'<table class="md">{body}</table>')
            rows.clear()

    for line in text.split("\n"):
        if line.strip().startswith("|"):
            flush_block()
            cells = _cells(line)
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        else:
            flush_rows()
            block.append(line)
    flush_rows()
    flush_block()
    return "\n".join(out)


def answer_body(a: dict) -> str:
    agent = a["system"] == "agent"
    html_text = mark_quotes(md_with_tables(a["text"]), a["log"], agent)
    if agent:
        html_text = restore_scars(html_text, a["log"])
    return f'<div class="body"{dir_attr(a["text"])}>{html_text}</div>'


def quotation_list(a: dict) -> str:
    if not a["log"]:
        return '<p class="empty">No quotations were checked in this answer.</p>'
    agent = a["system"] == "agent"
    rows = []
    for n, e in enumerate(a["log"], 1):
        ok = bool(e.get("verified"))
        verdict = ("verified" if ok else "removed by the gate") if agent else \
                  ("found in corpus" if ok else "not in this corpus")
        badges = " ".join(f'<span class="badge">{esc(label)}</span>'
                          for key, (label, _) in ADVISORY.items()
                          if key != "in_session" and e.get(key))
        found = ", ".join(e.get("found_in") or []) or "—"
        rows.append(f'<tr class="qrow {"ok" if ok else "no"}"><td class="n">{n}</td>'
                    f'<td{dir_attr(e.get("quote", ""))}>{esc(e.get("quote", ""))}</td>'
                    f'<td class="verdict">{verdict} {badges}</td>'
                    f'<td class="src">{esc(found)}</td></tr>')
    return ('<table class="qlist"><thead><tr><th>#</th><th>quotation</th><th>verdict</th>'
            f'<th>found in</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _call(c: dict) -> str:
    args = "".join(f' <span class="arg"><span class="k">{esc(k)}=</span>'
                   f'<span{dir_attr(str(v))}>{esc(v)}</span></span>'
                   for k, v in (c.get("args") or {}).items())
    return f'<li><code>{esc(c.get("tool"))}</code>{args}</li>'


def compact_trail(calls: list) -> str:
    if not calls:
        return ""
    parts = []
    for stage in STAGES:
        staged = [c for c in calls if c.get("stage") == stage]
        if staged:
            parts.append(f'<h5>{esc(STAGE_TITLE[stage])} · {len(staged)} calls</h5>'
                         f'<ol class="trail">{"".join(_call(c) for c in staged)}</ol>')
    other = [c for c in calls if c.get("stage") not in STAGES]
    if other:
        parts.append(f'<h5>Unlabelled · {len(other)} calls</h5>'
                     f'<ol class="trail">{"".join(_call(c) for c in other)}</ol>')
    return "".join(parts)


TITLE = "Query-Agent outputs: the full conference question set"
SUBTITLE = ("Companion to “Agentic AI and Human Reason in Hadith Scholarship”, "
            "Tübingen, 7–9 October 2026")
AUTHOR = "Muhammed Ragab"
AFFIL = "Leibniz-Institute for Educational Media | Georg-Eckert-Institute"
EMAIL = os.environ.get("COMPANION_CONTACT", "")

SYSTEM_ABOUT = [
    "The Query-Agent answers a question about ḥadīth from inside one classical "
    "commentary tradition — al-ʿAynī on al-Bukhārī, al-Nawawī on Muslim, or "
    "al-Māzandarānī on al-Kāfī — reading only that commentary, its ḥadīth collection "
    "and shared reference works.",
    "It works in three stages (isnād, matn, synthesis); at each stage a language model "
    "(Qwen3.8-27B, temperature 0) calls tools that search and retrieve passages, and "
    "every call is logged.",
    "Before an answer is stored, every quotation in it is checked against the corpus, "
    "and quotations that cannot be found are removed.",
]

BATCH = ("This document prints every answer of the final run of the conference question "
         "set: 26 Query-Agent answers (one run per question and commentator, "
         "24 and 27 September 2026) and 15 answers from a frontier model (OpenAI "
         "GPT-5.6 Luna, one plain call per question, no sources or tools). Nothing has "
         "been selected out, and no answer text has been edited.")

DISCLAIMER = ("Research outputs of an experimental system. "
              "Not a fatwa and not a religious ruling.")
DISCLAIMER_MORE = [
    "Each answer is a single run; another run of the same question can differ.",
    "All review gates were approved automatically. No answer was reviewed by a ḥadīth "
    "specialist.",
    "“Not verified” means “not found in this corpus”. It does not mean fabricated: a "
    "quotation can be genuine and still lie outside the files the system searches.",
]

HOW_TO_READ = [
    ('<span class="q ok">green text</span><sup>[n✓]</sup>',
     "a quotation found verbatim in the corpus; n is its row in the quotation list below "
     "the answer"),
    ('<span class="stripped"><s>struck text</s></span>',
     "a quotation the system removed before storing the answer, because it could not be "
     "found; printed here so the removal is visible"),
    ('<span class="q no">amber text</span><sup>[n✗]</sup>',
     "(Luna only) a quotation not found in this corpus; nothing was removed from Luna’s "
     "answers"),
    ("Quotation list", "every quotation checked, with the verdict and the corpus file it "
     "was found in. Quotations whose wording in the answer differs slightly from the "
     "checked string are listed but not marked inline"),
    ("Tool trail", "every tool call the agent made, by stage, with its arguments"),
]

CHAPTER_TITLES = {
    "q1": "ra’y and qiyās", "q2": "conflicting ḥadīths", "q3": "riwāya bi-l-maʿnā",
    "q4": "jabr and qadar", "q5": "taqiyya", "q6": "unanswered duʿāʾ",
    "q7": "the heirs of the Prophet’s knowledge", "q8": "«اطلبوا العلم ولو بالصين»",
    "q9": "al-šuʾm", "q10": "«يقطع الصلاة المرأة والحمار والكلب»",
    "q11": "«أُمرت أن أقاتل الناس»",
}
COMMENTATOR_NAMES = {
    "cayni": "al-ʿAynī · ʿUmdat al-Qārī (on al-Bukhārī)",
    "nawawi": "al-Nawawī · al-Minhāj (on Muslim)",
    "mazandarani": "al-Māzandarānī · Sharḥ Uṣūl al-Kāfī",
}
VARIANT_NAMES = {"ar": "Arabic", "en": "English", "colloq": "Egyptian colloquial"}

# Shown in the talk (slides/v3/hadith_talk_v3.tex). Keys are companion_data keys.
IN_TALK = {
    "set_q1_ray_cayni": "slide “Level 1, q1: ra’y and qiyās”",
    "set_q1_ray_mazandarani": "slides “Level 1, q1” and “Ask in English”",
    "set_q1_ray_en_mazandarani": "slide “Ask in English: the sources stay Arabic”",
    "set_q7_warith_mazandarani": "slide «تثبّت» (viewer screenshot) and backup “What it declines to claim”",
    "set_q10_salat_cayni": "slides “q10, what the agent did”, “Anatomy of an answer”, «تثبّت», "
                           "“agent vs Luna”; full synthesis in the backup",
    "set_q10_salat_mazandarani": "backup “q10, two answers” and “the underlying quotations”",
    "set_q10_salat_en_mazandarani": "slides “Ask in English” and “Limits, and what comes next”",
    "set_q9_shum_mazandarani": "backup “What it declines to claim”; slide “Limits, and what comes next” (missed narrations)",
    "set_q5_taqiyya_mazandarani": "backup “invariance across register”",
    "set_q5_taqiyya_en_mazandarani": "backup “invariance across register”",
    "set_q5_taqiyya_colloq_mazandarani": "backup “invariance across register”",
    "luna_q10_salat": "slide “Same ḥadīth, same question: agent vs Luna”",
}

EDITOR_NOTES = {
    "set_q11_umirtu_cayni":
        "The stored answer breaks off mid-sentence («حتى …). The run's output limit "
        "was 4,000 tokens; the answer is printed as stored, not completed.",
    "set_q10_salat_en_mazandarani":
        "A weak run, included deliberately: 3 verified quotations (4 checked), and none of the "
        "counter-narration the Arabic run of the same question found.",
    "set_q1_ray_mazandarani":
        "The Arabic and English runs of q1 were rewritten into differently worded "
        "canonical questions (see “searched as” in each header), so they are not the "
        "same question twice.",
    "set_q1_ray_en_mazandarani":
        "The Arabic and English runs of q1 were rewritten into differently worded "
        "canonical questions (see “searched as” in each header), so they are not the "
        "same question twice.",
}


def header(a: dict) -> str:
    if a["system"] == "agent":
        who = COMMENTATOR_NAMES[a["commentator"]]
        model = (f'{esc(a["model_id"])} via {esc(a["provider"])}/{esc(a["provider_route"])}, '
                 f'{esc(a["quantization"])}, temperature {esc(a["temperature"])}')
        status = "Gates auto-approved · not reviewed by a ḥadīth specialist"
    else:
        who = "Frontier baseline · OpenAI GPT-5.6 Luna"
        model = (f'{esc(a["model_id"])} via {esc(a["provider"])} (route: {esc(a["provider_route"])}), '
                 f'max_tokens {esc(a["max_tokens"])}, one plain call, no sources, no tools')
        status = "Not reviewed by a ḥadīth specialist"
    rows = [("Question", f'<span{dir_attr(a["question"])}>{esc(a["question"])}</span>')]
    if a["question_ar"]:
        terms = "، ".join(a["search_terms"])
        rows.append(("Searched as", f'<span dir="rtl" class="rtl">{esc(a["question_ar"])}</span>'
                     + (f'<br><span dir="rtl" class="rtl terms">{esc(terms)}</span>' if terms else "")))
    rows += [("System", esc(who)), ("Model", model),
             ("Run", f'{esc(a["date"])} · one run · <code>{esc(a["run_file"])}</code>'),
             ("Status", status),
             ("Quotations", f'{a["verified"]} of {a["quotations"]} verified'
                            + (f' ({a["echoes"]} echo the question)' if a["echoes"] else ""))]
    if a["key"] in IN_TALK:
        rows.append(("In the talk", esc(IN_TALK[a["key"]])))
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    variant = VARIANT_NAMES[a["variant"]]
    title = (f'{who.split(" · ")[0]}' if a["system"] == "agent" else "Luna") + f" · {variant}"
    return (f'<h3 class="answer-title" id="{a["aid"]}"><span class="aid">[{a["aid"]}]</span> '
            f'{esc(title)}</h3><table class="hdr">{body}</table>')


ARABIC_RUN = re.compile("«?[" + chr(0x600) + "-" + chr(0x6ff) + "]+(?: [" + chr(0x600) + "-" + chr(0x6ff) + "]+)*")


def isolate_arabic(html_text: str) -> str:
    """Wrap Arabic runs in an English sentence so they keep their own order and font."""
    return ARABIC_RUN.sub(lambda m: f'<bdi class="ar">{m.group(0)}</bdi>', html_text)


def answer_section(a: dict) -> str:
    note = EDITOR_NOTES.get(a["key"])
    note_html = (f'<aside class="note"><strong>Editor’s note.</strong> {isolate_arabic(esc(note))}</aside>'
                 if note else "")
    trail = (f'<h4>Tool trail · {len(a["calls"])} calls</h4>{compact_trail(a["calls"])}'
             if a["system"] == "agent" else
             '<p class="empty">No tool trail: one plain call without tools.</p>')
    return (f'<section class="answer">{header(a)}{note_html}'
            f'<h4>Answer</h4>{answer_body(a)}'
            f'<h4>Quotations checked</h4>{quotation_list(a)}{trail}</section>')


def word_count(a: dict) -> int:
    return len(a["text"].split())


def index_table(answers: list, pages: dict) -> str:
    rows = "".join(
        f'<tr><td>{a["aid"]}</td><td>{a["chapter"]}</td><td>{VARIANT_NAMES[a["variant"]]}</td>'
        f'<td>{esc(COMMENTATOR_NAMES[a["commentator"]].split(" · ")[0] if a["commentator"] else "Luna")}</td>'
        f'<td class="num">{a["verified"]} of {a["quotations"]}</td>'
        f'<td class="num">{a["echoes"]}</td><td class="num">{word_count(a)}</td>'
        f'<td class="num">{len(a["calls"]) if a["system"] == "agent" else "—"}</td>'
        f'<td class="num pg" data-aid="{a["aid"]}">{pages.get(a["aid"], "000")}</td></tr>'
        for a in answers)
    note = ('<p class="small">Counts are taken from each run’s grounding log as recorded: '
            'every quotation checked, repeats included. The level-2 table in the project’s '
            'RESULTS.md leaves out quotations that only echo the question, so some of its '
            'numbers are lower. Words: whitespace-separated tokens of the stored answer.</p>')
    return (note + '<table class="index"><thead><tr><th>id</th><th>q</th><th>language</th>'
            '<th>system</th><th>verified</th><th>echoes q.</th><th>words</th>'
            f'<th>tool calls</th><th>page</th></tr></thead><tbody>{rows}</tbody></table>')


def ch_title(title: str) -> str:
    """Chapter titles may quote Arabic; isolate it so it keeps its own font and order."""
    if any("؀" <= c <= "ۿ" for c in title):
        return f'<bdi class="ar">{esc(title)}</bdi>'
    return esc(title)


def toc(answers: list, pages: dict) -> str:
    items, chapter = [], None
    for a in answers:
        if a["chapter"] != chapter:
            chapter = a["chapter"]
            items.append(f'<li class="ch"><span>{chapter} · {ch_title(CHAPTER_TITLES[chapter])}</span></li>')
        who = COMMENTATOR_NAMES[a["commentator"]].split(" · ")[0] if a["commentator"] else "Luna"
        items.append(f'<li><span>[{a["aid"]}] {esc(who)} · {VARIANT_NAMES[a["variant"]]}</span>'
                     f'<span class="pg" data-aid="{a["aid"]}">{pages.get(a["aid"], "000")}</span></li>')
    return f'<ol class="toc">{"".join(items)}</ol>'


def build_html(answers: list, luna_system: str, pages: dict | None = None) -> str:
    pages = pages or {}
    chapters, chapter, level = [], None, None
    for a in answers:
        if a["level"] != level:
            level = a["level"]
            chapters.append(f'<h1 class="part">Part {"I" if level == 1 else "II"} · '
                            f'{"Level 1: a tradition asked its own questions" if level == 1 else "Level 2: a problematic ḥadīth under an objection from outside"}</h1>')
            if level == 1:
                chapters.append(f'<p class="small">Luna’s system prompt for every answer: '
                                f'“{esc(luna_system)}”</p>')
        if a["chapter"] != chapter:
            chapter = a["chapter"]
            chapters.append(f'<h2 class="chapter">{chapter} · {ch_title(CHAPTER_TITLES[chapter])}</h2>'
                            f'<p class="orig"{dir_attr(a["question"])}>{esc(a["question"])}</p>')
        chapters.append(answer_section(a))
    about = "".join(f"<p>{esc(s)}</p>" for s in SYSTEM_ABOUT)
    how = "".join(f"<tr><td>{k}</td><td>{esc(v)}</td></tr>" for k, v in HOW_TO_READ)
    more = "".join(f"<li>{esc(s)}</li>" for s in DISCLAIMER_MORE)
    return f"""<!DOCTYPE html>
<html lang="en"><meta charset="utf-8"><title>{esc(TITLE)}</title>
<style>{CSS}</style>
<section class="cover">
  <h1>{esc(TITLE)}</h1><p class="sub">{esc(SUBTITLE)}</p>
  <p class="author">{esc(AUTHOR)}<br>{esc(AFFIL)}<br>{esc(EMAIL)}</p>
  <h2>What the system is</h2>{about}
  <h2>What is in this document</h2><p>{esc(BATCH)}</p>
  <h2>How to read the marks</h2><table class="how">{how}</table>
</section>
<section class="disclaimer"><p class="big">{esc(DISCLAIMER)}</p><ul>{more}</ul></section>
<section class="contents"><h1>Contents</h1>{toc(answers, pages)}</section>
{"".join(chapters)}
<section class="index-sec"><h1>Index of all answers</h1>{index_table(answers, pages)}</section>
</html>"""


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; }
* { box-sizing: border-box; }
body { font-family: "Noto Serif", "Linux Libertine G", "Cambria", serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
.rtl, [dir="rtl"] { font-family: "Amiri", "Noto Naskh Arabic", "Scheherazade", serif; font-size: 1.12em; line-height: 1.9; text-align: right; }
h1 { font-size: 20pt; } h1.part { page-break-before: always; margin-top: 30%; }
h2.chapter { page-break-before: always; font-size: 16pt; border-bottom: 2px solid #1a1a1a; }
.orig { font-size: 1.1em; background: #f4f4f1; padding: .6em .9em; }
section.answer { page-break-before: always; }
h2.chapter + .orig + section.answer { page-break-before: avoid; }
h3.answer-title { font-size: 13pt; margin: 0 0 .4em; } .aid { color: #666; font-weight: 400; }
table.hdr { width: 100%; border-collapse: collapse; font-size: 9pt; margin-bottom: .8em; }
table.hdr th { text-align: left; width: 7.5em; color: #666; font-weight: 400; vertical-align: top; padding: .15em .4em .15em 0; }
table.hdr td { padding: .15em 0; }
.terms { color: #555; font-size: .95em; }
h4 { font-size: 10pt; text-transform: uppercase; letter-spacing: .06em; color: #2c4c7c; margin: 1.2em 0 .4em; }
h5 { font-size: 9pt; margin: .6em 0 .2em; color: #2c4c7c; }
.note { border-left: 3px solid #8a6100; background: #fbf6ea; padding: .5em .8em; font-size: 9pt; margin: .6em 0; }
.body blockquote { border-inline-start: 3px solid #ddd; margin-inline-start: 0; padding-inline-start: .8em; }
.q.ok { background: #e6f4ea; } .q.no { background: #fff1d6; }
sup.ann { font-size: 7pt; color: #1a7f43; font-family: "Noto Serif", "Linux Libertine G", "Cambria", serif; }
.stripped { background: #fdeceb; } .stripped s { color: #b3261e; }
.why { font-size: 7.5pt; color: #b3261e; } .rescued { background: #eaf6ee; } .why-ok { font-size: 7.5pt; color: #1a7f43; }
table.qlist { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
table.qlist th, table.qlist td { border-bottom: 1px solid #e3e3e3; padding: .2em .3em; vertical-align: top; }
table.qlist td.n { width: 2em; color: #666; } table.qlist td.verdict { width: 9em; } table.qlist td.src { width: 12em; font-family: ui-monospace, monospace; font-size: 7.5pt; color: #555; word-break: break-all; }
tr.qrow.no td.verdict { color: #b3261e; } tr.qrow.ok td.verdict { color: #1a7f43; }
.badge { border: 1px solid #8a6100; color: #8a6100; border-radius: 99px; padding: 0 .4em; font-size: 7pt; white-space: nowrap; }
ol.trail { font-size: 8pt; margin: .2em 0 .5em; padding-left: 2em; } ol.trail li { margin: .1em 0; }
ol.trail code { background: #eef2f8; color: #2c4c7c; padding: 0 .3em; }
.arg .k { color: #666; }
.empty { color: #666; font-style: italic; }
.cover h1 { font-size: 22pt; margin: 0 0 .3em; } .cover .sub { font-size: 12pt; color: #444; }
.cover .author { margin: 1em 0; } .cover h2 { font-size: 11.5pt; margin: .9em 0 .3em; } .cover p { margin: .3em 0; }
table.how td { padding: .2em .6em .2em 0; vertical-align: top; font-size: 9pt; }
table.how td:first-child { white-space: nowrap; }
.disclaimer { page-break-before: always; padding-top: 25%; }
.disclaimer .big { font-size: 18pt; font-weight: 700; }
.contents { page-break-before: always; }
ol.toc { list-style: none; padding: 0; font-size: 9.5pt; }
ol.toc li { display: flex; justify-content: space-between; border-bottom: 1px dotted #ccc; }
ol.toc li.ch { font-weight: 700; border: 0; margin-top: .6em; }
.index-sec { page-break-before: always; }
table.index { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
table.index th, table.index td { border-bottom: 1px solid #e3e3e3; padding: .15em .3em; text-align: left; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.small { font-size: 8.5pt; color: #555; }
sup.ann, .why, .why-ok, .badge { unicode-bidi: isolate; direction: ltr; }
bdi.ar { font-family: "Amiri", "Noto Naskh Arabic", "Scheherazade", serif; font-size: 1.1em; }
table.md { border-collapse: collapse; margin: .6em 0; font-size: 9pt; }
table.md td { border: 1px solid #ddd; padding: .15em .5em; }
code { font-size: .9em; }
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the companion HTML.")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("-o", "--out", type=Path, default=Path("companion/query_agent_outputs.html"))
    args = p.parse_args(argv)
    answers = cd.load_batch(args.outputs)
    _, system = cd.load_luna(args.outputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(answers, system), encoding="utf-8")
    print(args.out, len(answers), "answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

