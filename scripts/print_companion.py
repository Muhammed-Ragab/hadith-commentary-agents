"""Print the companion HTML to PDF, fill in page numbers, stamp folios, add bookmarks.

Chrome cannot compute page numbers for a table of contents, so the page of each answer
is read back from a first print and the HTML rebuilt with it. Pagination does not
depend on the numbers (TOC rows are fixed-height flex lines), and the second print is
checked to prove it.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import build_companion as bc
from scripts import companion_data as cd

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path.resolve()}",
                    html_path.resolve().as_uri()], check=True, capture_output=True, timeout=300)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Chrome produced no PDF at {pdf_path}")


def answer_pages(pdf_path: Path, aids: list) -> dict:
    """1-based page of each answer: the second page whose text holds [Axx] (first = TOC)."""
    doc = fitz.open(pdf_path)
    seen = {aid: [] for aid in aids}
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        for aid in re.findall(r"\[(A\d{2})\]", text):
            if aid in seen and (not seen[aid] or seen[aid][-1] != i):
                seen[aid].append(i)
    doc.close()
    bad = {aid: p for aid, p in seen.items() if len(p) != 2}
    if bad:
        raise RuntimeError(f"cannot locate answers unambiguously: {bad}")
    return {aid: p[1] for aid, p in seen.items()}


def finish(pdf_in: Path, pdf_out: Path, answers: list, pages: dict) -> None:
    doc = fitz.open(pdf_in)
    for i, page in enumerate(doc, 1):
        if i == 1:
            continue
        r = page.rect
        page.insert_text((r.width / 2 - 6, r.height - 22), str(i), fontsize=8, fontname="helv")
    toc, chapter = [], None
    for a in answers:
        if a["chapter"] != chapter:
            chapter = a["chapter"]
            toc.append([1, f'{chapter} · {bc.CHAPTER_TITLES[chapter]}', pages[a["aid"]]])
        who = bc.COMMENTATOR_NAMES[a["commentator"]].split(" · ")[0] if a["commentator"] else "Luna"
        toc.append([2, f'[{a["aid"]}] {who} · {bc.VARIANT_NAMES[a["variant"]]}', pages[a["aid"]]])
    doc.set_toc(toc)
    doc.save(pdf_out, garbage=3, deflate=True)
    doc.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Print the companion PDF.")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--dir", type=Path, default=Path("companion"))
    args = p.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)
    answers = cd.load_batch(args.outputs)
    _, system = cd.load_luna(args.outputs)
    html_path, draft = args.dir / "query_agent_outputs.html", args.dir / "_draft.pdf"
    aids = [a["aid"] for a in answers]

    html_path.write_text(bc.build_html(answers, system), encoding="utf-8")
    chrome_pdf(html_path, draft)
    pages = answer_pages(draft, aids)
    for _ in range(2):
        html_path.write_text(bc.build_html(answers, system, pages), encoding="utf-8")
        chrome_pdf(html_path, draft)
        again = answer_pages(draft, aids)
        if again == pages:
            break
        pages = again
    else:
        raise RuntimeError("pagination did not settle after filling page numbers")
    out = args.dir / "query_agent_outputs.pdf"
    finish(draft, out, answers, pages)
    draft.unlink()
    print(out, f"{fitz.open(out).page_count} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
