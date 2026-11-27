"""Corpus statistics guard. See spec §3.0."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "babs"

BABS_FILES = {
    "nawawi": DATA_DIR / "nawawi_babs.json",
    "cayni": DATA_DIR / "cayni_babs.json",
    # Shia arm (handoff.md §10.5). Mirʾāt al-ʿUqūl is not in OpenITI; Māzandarānī's
    # Sharḥ Uṣūl al-Kāfī is, and comments on al-Kāfī, which is the inquiry corpus.
    "mazandarani": DATA_DIR / "mazandarani_babs.json",
}

# Import this rather than re-typing the tuple; adding a commentator should be one edit.
COMMENTATORS = tuple(BABS_FILES)


MUQADDIMA = "مقدمة"


def is_bab(unit: dict) -> bool:
    """A bāb-level unit, as opposed to front matter or a whole undivided kitāb.

    Two kinds of non-bāb slip out of the al-Kāfī parse, both only in Māzandarānī:

    * the muqaddima — 170,744 characters of OpenITI `#META#` header plus front
      matter at index 0, 26× the median bāb;
    * six units where the parser found no bāb markers inside a kitāb and emitted the
      whole kitāb (`bab == kitab`), including al-Rawḍa at 1,692,380 characters.

    Every one-bāb preflight to date segmented the muqaddima as the Shia arm, so the
    arms were never matched at index 0. Filtering here rather than at the call sites
    is deliberate: `run_naive_batch.py` shards the full corpus with no `--limit`, and
    its `bab_ref` indices must agree with `run_segmentation.py`'s. One source of
    truth, or the phases silently disagree about which unit `mazandarani/0` is.
    """
    return unit["bab"] != MUQADDIMA and not (unit["kitab"] and unit["bab"] == unit["kitab"])


def load_babs(commentator: str) -> list[dict]:
    """Load the bāb-level units for one commentator. See `is_bab`."""
    babs = json.loads(BABS_FILES[commentator].read_text(encoding="utf-8"))
    return [b for b in babs if is_bab(b)]


def parse_sample(spec: str) -> tuple[int, int, int]:
    """Read a `KITABS:PER_KITAB:SEED` sample spec, e.g. "12:5:20260914"."""
    parts = (spec or "").split(":")
    if len(parts) != 3:
        raise ValueError(
            f"sample spec {spec!r} must be KITABS:PER_KITAB:SEED, e.g. 12:5:20260914"
        )
    try:
        n_kitabs, per_kitab, seed = (int(part) for part in parts)
    except ValueError:
        raise ValueError(f"sample spec {spec!r} must be three integers") from None
    if n_kitabs < 1 or per_kitab < 1:
        raise ValueError(f"sample spec {spec!r} needs at least 1 kitāb and 1 bāb")
    return n_kitabs, per_kitab, seed


def sample_indices(babs: list[dict], n_kitabs: int, per_kitab: int,
                   seed: int) -> list[int]:
    """Global positions of a stratified sample: `per_kitab` consecutive bābs from each
    of `n_kitabs` kitābs.

    `--limit N` means indices 0..N-1, i.e. the opening kitābs only -- and §8.17 showed
    the opening is the least representative part of the corpus. Spread across kitābs
    answers "did you read anything but the first chapter?"; consecutive *within* a
    kitāb preserves the two things that are kitāb-scoped, the scholar-review sitting
    and the accumulating method hypothesis.

    The window start inside a kitāb is drawn too. Always taking a kitāb's first bābs
    would reproduce the same opening bias one level down.

    Returns global indices because `bab_ref` is `name/<position in load_babs>`, and
    segmentation, the naive pre-pass and the study graph all join on it.
    """
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for index, bab in enumerate(babs):
        kitab = bab.get("kitab", "")
        # Nawawi's units 0-4 are real babs that precede the first kitab header, so they
        # share an empty kitab name: a nameless stratum and a nameless review sheet.
        # Skipped here rather than in load_babs, which would renumber every bab_ref.
        if not kitab.strip():
            continue
        if kitab not in members:
            members[kitab] = []
            order.append(kitab)
        members[kitab].append(index)

    rng = random.Random(seed)
    # Sample from a list in corpus order so the draw cannot depend on dict iteration.
    chosen = sorted(rng.sample(order, min(n_kitabs, len(order))), key=order.index)

    picked: list[int] = []
    for kitab in chosen:
        indices = members[kitab]
        # A short kitāb contributes what it has: demanding per_kitab would quietly
        # restrict the study to large kitābs, which is a bias rather than a filter.
        take = min(per_kitab, len(indices))
        start = rng.randrange(len(indices) - take + 1)
        picked.extend(indices[start:start + take])
    return sorted(picked)


def corpus_stats(commentator: str) -> dict:
    """Return {units, kitabs, chars} for one commentator."""
    babs = load_babs(commentator)
    return {
        "units": len(babs),
        "kitabs": len(Counter(b["kitab"] for b in babs)),
        "chars": sum(len(b["text"]) for b in babs),
    }


if __name__ == "__main__":
    total = 0
    for name in BABS_FILES:
        s = corpus_stats(name)
        total += s["units"]
        print(f"{name:8} units={s['units']:5} kitabs={s['kitabs']:3} chars={s['chars']:,}")
    print(f"{'TOTAL':8} units={total:5}")
    sys.exit(0)
