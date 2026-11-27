"""Fail a matched run unless every arm is complete and its evidence is usable."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics.grounding import is_grounded
from scripts.check_corpus import (COMMENTATORS, load_babs, parse_sample,
                                  sample_indices)
from scripts.run_segmentation import reject_reason

_ARABIC = re.compile(r"[\u0600-\u06ff]")

# B4 made an unsegmentable b\u0101b a declared shortfall rather than a dead arm. Without a
# ceiling that also passes an arm which produced nothing at all, as long as it logged a
# failure for every b\u0101b -- the completeness guard would be gone above the "declared"
# bar rather than relaxed, and \u00a78.14's rule stands: a red arm beats a fabricated
# finding.
#
# The allowance is absolute OR proportional, whichever is larger. A pure rate makes one
# failure of a two-b\u0101b smoke run a 50% catastrophe, and the known cause is a fixed small
# number of pathological b\u0101bs -- 2 of \u02bfAyn\u012b's 3,959 cannot be windowed (\u00a78.17) -- rather
# than a fraction of any sample.
MAX_DECLARED_FAILURE_RATE = 0.10
MIN_DECLARED_FAILURES = 2


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_jsonl_glob(directory: Path, pattern: str) -> list[dict]:
    return [rec for path in sorted(directory.glob(pattern)) for rec in _read_jsonl(path)]


def validate_arm(root: Path, name: str, expected_babs: int,
                 sources: dict[str, str]) -> list[str]:
    arm = Path(root) / name
    notes = _read_jsonl(arm / f"{name}_notes.jsonl")
    failures = _read_jsonl(arm / f"{name}_failures.jsonl")
    errors = []

    for record in _read_jsonl_glob(Path(root) / "segmentation", f"{name}_seg.*.jsonl"):
        ref = record.get("bab_ref", "")
        reason = reject_reason(
            int(record.get("matn_end_char", 0)),
            float(record.get("confidence", 0.0)),
            sources.get(ref, ""),
            start=int(record.get("matn_start_char", 0)),
        )
        if reason:
            errors.append(f"{ref}: segmentation boundary is not credible -- {reason}")

    method = arm / f"{name}_method.md"
    if not method.is_file() or not method.read_text(encoding="utf-8").strip():
        errors.append(f"{name}: missing method file")

    # B4: a bāb segmentation could not window, or that horizon_diff gave up on
    # twice, is now a DECLARED shortfall rather than a fatal one -- run_segmentation
    # and horizon_diff (hermeneutics/nodes.py::_write_failure) both log it instead
    # of writing a note, and the graph skips it rather than crash (nodes.py's
    # select_next_bab). What must never happen is a ref missing from `notes` with
    # NO matching failure record: that is a silent gap, not a declared one.
    #
    # `sources` is exactly what main() requested for this arm -- built from
    # load_babs()[:expected_babs] today, but comparing against it rather than
    # rebuilding range(expected_babs) here keeps this correct once stratified
    # sampling (S1) makes the requested indices non-contiguous too.
    expected_refs = set(sources)
    actual_refs = {record.get("bab_ref") for record in notes}
    seg_failures = _read_jsonl_glob(Path(root) / "segmentation", f"{name}_seg_failures.*.jsonl")
    failure_refs = {record.get("bab_ref") for record in failures + seg_failures}

    missing = expected_refs - actual_refs - failure_refs
    if missing:
        errors.append(
            f"{name}: {len(missing)} requested bāb(s) have neither a note nor a "
            f"declared failure -- {sorted(missing)}"
        )
    extra = actual_refs - expected_refs
    if extra:
        errors.append(f"{name}: notes reference babs outside the requested set -- {sorted(extra)}")
    # A bāb with a note is COVERED, whatever else it logged. B4 widened _write_failure
    # from segmentation-only to every failure, so a move-level grounding rejection on a
    # bāb that went on to produce a perfectly good note now also writes a failure record
    # (cayni/22). Treating that as an error reported a complete arm as broken, and
    # counting it as a declared shortfall pushed a healthy arm toward the ceiling.
    # It is a move-rejection: reported, not fatal, and not a shortfall.
    move_rejections = actual_refs & failure_refs
    if move_rejections:
        print(f"{name}: {len(move_rejections)} move-rejection(s) on bābs that still "
              f"produced a note -- {sorted(move_rejections)}")

    # Only bābs that produced NO note are shortfalls, so only they count toward the
    # ceiling and the shortfall line below.
    shortfall_refs = failure_refs - actual_refs

    # Printed unconditionally when there IS a shortfall, pass or fail -- a silent
    # 58/60 is exactly what the declared-shortfall rule must not become (task B4).
    if shortfall_refs:
        print(f"{name}: {len(actual_refs)}/{len(expected_refs)} notes, "
              f"{len(shortfall_refs)} declared failure(s)")
        allowed = max(MIN_DECLARED_FAILURES,
                      math.ceil(MAX_DECLARED_FAILURE_RATE * len(expected_refs)))
        if len(shortfall_refs) > allowed:
            errors.append(
                f"{name}: {len(shortfall_refs)} declared failure(s) of "
                f"{len(expected_refs)} requested exceeds the ceiling of {allowed} -- "
                f"this is a failed arm, not a shortfall"
            )

    for record in notes:
        ref = record.get("bab_ref", "")
        moves = record.get("moves", [])
        status = record.get("status", "moves")
        if status == "no_move":
            if moves:
                errors.append(f"{ref}: no_move record contains moves")
            if not str(record.get("no_move_reason", "")).strip():
                errors.append(f"{ref}: no_move record has no reason")
            continue
        if status != "moves":
            errors.append(f"{ref}: unknown status {status!r}")
            continue
        if not moves:
            errors.append(f"{ref}: no interpretive moves")
        for index, move in enumerate(moves):
            evidence = move.get("arabic_evidence", "")
            if not _ARABIC.search(evidence):
                errors.append(f"{ref} move {index}: evidence is not Arabic")
            if not is_grounded(evidence, sources.get(ref, "")):
                errors.append(f"{ref} move {index}: evidence is not grounded")
    return errors


def sample_specs(values: list[str]) -> dict[str, str]:
    """Resolve `--sample` values into one spec per arm.

    A bare `KITABS:PER_KITAB:SEED` applies to every arm; `ARM=SPEC` overrides one. The
    arms genuinely need different specs -- Māzandarānī has 5 bāb-bearing kitābs against
    ʿAynī's 87 -- so a single spec cannot give all three a comparable sample size.

    Every arm must end up with a spec: one left out would be validated against the
    whole corpus and report every sampled bāb as unrequested.
    """
    default = None
    per_arm: dict[str, str] = {}
    for value in values:
        arm, sep, spec = value.partition("=")
        if not sep:
            default = value
            continue
        if arm not in COMMENTATORS:
            raise ValueError(f"unknown arm {arm!r} in --sample; expected one of "
                             + ", ".join(COMMENTATORS))
        per_arm[arm] = spec
    specs = {name: per_arm.get(name, default) for name in COMMENTATORS}
    missing = sorted(name for name, spec in specs.items() if not spec)
    if missing:
        raise ValueError(
            "no --sample spec for " + ", ".join(missing)
            + "; pass a bare SPEC as the default or ARM=SPEC for each arm"
        )
    for name, spec in specs.items():
        parse_sample(spec)  # fail here, not three phases later
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-babs", type=int, default=None,
                        help="the first N babs; use --sample for a stratified run")
    parser.add_argument("--sample", action="append", default=None,
                        help="the SAME spec the run was given: KITABS:PER_KITAB:SEED "
                             "for every arm, or ARM=SPEC (repeatable) per arm")
    args = parser.parse_args()
    if not args.sample and args.expected_babs is None:
        parser.error("pass --expected-babs or --sample")
    specs = sample_specs(args.sample) if args.sample else {}

    errors = []
    for name in COMMENTATORS:
        babs = load_babs(name)
        # `sources` defines which babs were REQUESTED -- validate_arm compares note
        # refs against exactly this set, so a stratified run validated against
        # range(expected_babs) would report every sampled bab as unrequested.
        if specs:
            picked = sample_indices(babs, *parse_sample(specs[name]))
        else:
            picked = list(range(min(args.expected_babs, len(babs))))
        sources = {f"{name}/{i}": babs[i]["text"] for i in picked}
        arm_errors = validate_arm(args.root, name, len(picked), sources)
        errors.extend(arm_errors)
        if not arm_errors:
            print(f"{name}: validated {len(picked)} babs")

    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
