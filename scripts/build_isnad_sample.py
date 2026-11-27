"""Rebuild the 500-item isnad sample notebook 01 (seed 42) drew. Spec §3.5, defect #9.

Reproduces notebook 01 cell 2's load_and_sample verbatim, as a runnable, auditable
script rather than a one-off command. hermeneutics/isnad_graph.py:load_sample reads
poc/data/<corpus>_sample.json -- this is what writes that file.

Two nesting shapes exist across the three corpora:
- bukhari, muslim: chapters -> sub_chapters -> hadiths
- kafi: chapters -> hadiths

Traversal order matters: random.sample on a list is deterministic only for a fixed
input order, so hadiths are walked in file order (chapter, then sub_chapter, then
hadith), exactly as the notebook does -- never re-ordered or filtered differently.
"""
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SAMPLE_SIZE = 500
RANDOM_SEED = 42

RAW_PATHS = {
    "bukhari": DATA_DIR / "bukhari_raw.json",
    "kafi": DATA_DIR / "kafi_raw.json",
    "muslim": DATA_DIR / "openiti" / "muslim_raw.json",
}

# Which corpora nest hadiths under chapters -> sub_chapters -> hadiths (bukhari-style)
# versus chapters -> hadiths (kafi-style). Verbatim split from notebook 01 cell 2.
NESTED_CORPORA = {"bukhari", "muslim"}


def load_and_sample(raw_path: Path, corpus_type: str, n: int = SAMPLE_SIZE,
                     seed: int = RANDOM_SEED) -> list[dict]:
    """Verbatim port of notebook 01 cell 2's load_and_sample."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_hadiths = []
    if corpus_type in NESTED_CORPORA:
        for chapter in raw.get("chapters", []):
            for sub in chapter.get("sub_chapters", []):
                for h in sub.get("hadiths", []):
                    if h.get("isnad") and h.get("matn"):
                        all_hadiths.append(h)
    else:  # kafi-style: chapters -> hadiths
        for chapter in raw.get("chapters", []):
            for h in chapter.get("hadiths", []):
                if h.get("isnad") and h.get("matn"):
                    all_hadiths.append(h)

    random.seed(seed)
    sample = random.sample(all_hadiths, min(n, len(all_hadiths)))
    return sample


def build_sample(corpus: str, size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> list[dict]:
    raw_path = RAW_PATHS[corpus]
    return load_and_sample(raw_path, corpus, n=size, seed=seed)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild an isnad sample by the exact procedure notebook 01 used.")
    parser.add_argument("--corpus", required=True, choices=sorted(RAW_PATHS))
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing sample file.")
    args = parser.parse_args(argv)

    out_path = DATA_DIR / f"{args.corpus}_sample.json"
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists and --force was not passed. Refusing to "
            "overwrite it: bukhari_sample.json and kafi_sample.json are inputs to "
            "already-frozen results in poc/RESULTS.md, and silently regenerating "
            "them would invalidate finished work. Pass --force if you are certain."
        )

    sample = build_sample(args.corpus, size=args.size, seed=args.seed)
    out_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.corpus}: wrote {len(sample)} hadiths to {out_path}")


if __name__ == "__main__":
    main()
