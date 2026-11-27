"""Score the shuffled-pair control. Spec §4.1.

PRE-REGISTERED DECISION RULE - set before any numbers were seen:

    if control_grounding >= 0.80 * real_grounding:
        the derived "method" reflects model priors rather than study,
        and Claim 1 is WITHDRAWN, not softened.

Costs zero model calls. Run it as soon as both arms finish.

Usage:
  python scripts/score_control.py nawawi
  python scripts/score_control.py nawawi --blind-sample outputs/reviews/blind.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics.grounding import grounding_rate, is_grounded
from scripts.check_corpus import COMMENTATORS, load_babs

ROOT = Path(__file__).resolve().parent.parent

# Pre-registered 2026-08-24 against evidence-grounding rate; RE-REGISTERED 2026-09-14
# against yield, before any control run existed and before any yield number for either
# arm had been seen. The bar is unchanged at 0.80 -- the metric was substituted, not
# the threshold moved. Disclose the substitution in the paper: the original metric was
# found structurally incapable of varying (see yield_rate's docstring), not inconvenient.
CLAIM1_WITHDRAWAL_THRESHOLD = 0.80


def load_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def failures_for(notes_path: Path) -> Path:
    """The failures file beside a notes file, named the way _write_failure names it.

    nodes.py:_write_failure derives the stem from `deps.notes_path`, so an arm running
    with `--control-notes <x>_matnonly_notes.jsonl` logs to `<x>_matnonly_failures.jsonl`.
    Hardcoding `<commentator>_control_failures.jsonl` here looked in the wrong place for
    any arm not named "control": the file was missing, `load_records` returned [], and
    the denominator lost every declared failure. That is one of the two reasons the
    matn-only arms scored on 59 instead of 65 (see RESULTS.md).
    """
    return notes_path.with_name(
        f"{notes_path.stem.removesuffix('_notes')}_failures.jsonl")


def yield_rate(records: list[dict], sources: dict[str, str], failures: int = 0,
               requested: int | None = None) -> float:
    """Grounded moves per REQUESTED bāb -- the control's primary measure.

    §4.1 made evidence-grounding rate primary on the premise that "a control unit
    grounds only if the model quotes the reassigned sharh". The model never wrote the
    evidence: `nodes.py` substituted the whole cited span, and since B5 the model's
    quote is verified against that span before the record is written. Either way the
    rate is 1.0 by construction -- 88/88, 76/76, 84/84 on run 15789363 -- so it cannot
    falsify anything.

    What varies is how much a model can PRODUCE when the sharh does not match the matn.
    Projecting priors onto mismatched commentary now ends in a retry, then a declared
    failure, or a `no_move`. `grounding_rate` counts written moves only, so it discards
    precisely those outcomes and scores the survivors at 1.0. Hence the denominator
    here is every bāb the arm was asked for: one that produced a record (moves or
    no_move) or a declared failure.
    """
    # `records + failures` only counts bābs the arm REACHED. A bāb skipped upstream --
    # a segmentation failure inherited via --segmentation-dir, or a run_naive_batch
    # sub-100-char skip -- produces neither, so it silently leaves the denominator and
    # inflates the yield. Pass the sample spec's count as `requested` to pin it.
    if requested is None:
        requested = len(records) + failures
    if not requested:
        return 0.0
    grounded = sum(
        1
        for record in records
        for move in record.get("moves", [])
        if is_grounded(move.get("arabic_evidence", ""),
                       sources.get(record.get("bab_ref", ""), ""))
    )
    return grounded / requested


def score(real: list[dict], control: list[dict], sources: dict[str, str],
          real_failures: int = 0, control_failures: int = 0,
          control_sources: dict[str, str] | None = None,
          requested: int | None = None) -> dict:
    """Both arms' yield plus the pre-registered verdict. Grounding is a diagnostic.

    The two arms MUST be scored against different source maps. The control's pairing
    reassigns each unit to another bāb's text; applying that same map to the study arm
    checks every real move against commentary the study never read, so `real_grounding`
    collapses to 0.0, `yield_ratio` falls out of 0/x, and the run reports that Claim 1
    survives for no reason connected to the data. That is the §4.1 trap `16c91fc` fixed
    on the control side, mirrored onto the real one. `control_sources` defaults to
    `sources` only so the identity-pairing unit tests can pass one map.
    """
    real_yield = yield_rate(real, sources, real_failures, requested)
    control_yield = yield_rate(control, control_sources or sources, control_failures,
                               requested)
    yield_ratio = control_yield / real_yield if real_yield else 0.0

    # Both arms run over the same sample, so one `requested` pins both. Reported so a
    # reader can see the gap rather than infer it from a yield that looks healthy.
    n_real_requested = requested if requested is not None else len(real) + real_failures
    n_control_requested = (requested if requested is not None
                           else len(control) + control_failures)

    return {
        "real_yield": round(real_yield, 4),
        "control_yield": round(control_yield, 4),
        "yield_ratio": round(yield_ratio, 4),
        # Reported, never decisive: 1.0 by construction in both arms. Keeping it in the
        # output makes the validation pass rate visible and documents why it is not the
        # verdict -- dropping it silently would look like the number was hidden.
        "real_grounding": round(grounding_rate(real, sources), 4),
        "control_grounding": round(grounding_rate(control, control_sources or sources), 4),
        "n_real_moves": sum(len(r.get("moves", [])) for r in real),
        "n_control_moves": sum(len(r.get("moves", [])) for r in control),
        "n_real_requested": n_real_requested,
        "n_control_requested": n_control_requested,
        # Requested minus reached. Nonzero means bābs were dropped before the arm saw
        # them; they belong in the denominator and are counted there.
        "n_real_unaccounted": n_real_requested - (len(real) + real_failures),
        "n_control_unaccounted": n_control_requested - (len(control) + control_failures),
        "threshold": CLAIM1_WITHDRAWAL_THRESHOLD,
        "metric": "grounded moves per requested bāb",
        "claim1_withdrawn": yield_ratio >= CLAIM1_WITHDRAWAL_THRESHOLD,
    }


def build_blind_sample(real: list[dict], control: list[dict], n_per_arm: int = 30,
                       seed: int = 20260821) -> list[dict]:
    """Spec §4.1 measure 2: shuffled, unlabelled records for scholar rating.

    n=60 supports a descriptive difference, not a powered significance test.
    Report it as such.
    """
    rng = random.Random(seed)
    picked = (rng.sample(real, min(n_per_arm, len(real)))
              + rng.sample(control, min(n_per_arm, len(control))))

    # Assign IDs before shuffling so they appear non-sequential in the output.
    indexed = list(enumerate(picked))
    rng.shuffle(indexed)

    return [
        {
            "id": i,
            "bab_ref": rec["bab_ref"],
            "gap": rec.get("gap", ""),
            "moves": rec.get("moves", []),
            "revised_reading": rec.get("revised_reading", ""),
            "rating": None,  # scholar fills 1-5 before unblinding
        }
        for i, rec in indexed
    ]


def commentary_sources(commentator: str,
                       control_pairing: dict[int, int]) -> tuple[dict[str, str], dict[str, str]]:
    """(real_sources, control_sources): bab_ref -> the text each arm actually read.

    Returns two maps, not one. They differ on exactly the paired bābs, and the control
    is a derangement over the study's own sample, so on a --sample run EVERY study bāb
    is a pairing target -- sharing one map scores the whole real arm against text it
    never saw. Both are built from a single load_babs(); ʿAynī is 43 MB.
    """
    babs = load_babs(commentator)
    real = {f"{commentator}/{i}": b["text"] for i, b in enumerate(babs)}
    control = dict(real)
    for target, source in control_pairing.items():
        control[f"{commentator}/{target}"] = babs[source]["text"]
    return real, control


def pairing_from_manifest(path: Path) -> dict[int, int]:
    """The pairing the control arm ACTUALLY ran, as run_control recorded it.

    Recomputing it here from `shuffle_within_kitab(load_babs(...))` reproduces the
    original only when the control ran over the whole corpus at the default fraction.
    Against a --sample run it silently returns a different map, every control record is
    then scored against the wrong source text, grounding collapses, and the ratio lands
    near zero -- reporting that Claim 1 survives, for no reason connected to the data.
    A pre-registered rule must not be decided by a reconstruction that can disagree
    with the run.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    pairing = manifest.get("control_pairing")
    if not pairing:
        raise SystemExit(f"{path} carries no control_pairing; is it the control run's "
                         "manifest.json?")
    return {int(k): int(v) for k, v in pairing.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="Score the shuffled-pair control.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--blind-sample", default=None, help="write a blinded rating file here")
    p.add_argument("--real-notes", default=None,
                   help="the study arm's <name>_notes.jsonl")
    p.add_argument("--control-notes", default=None,
                   help="the control arm's <name>_control_notes.jsonl")
    p.add_argument("--no-pairing", action="store_true",
                   help="score an arm that ran WITHOUT a pairing -- the matn-only arm. "
                        "Its quotes come from each bāb's own matn, which is a slice of "
                        "that bāb's own text, so both arms score against the same "
                        "unshuffled map and no manifest pairing is needed.")
    p.add_argument("--requested", type=int, default=None,
                   help="bābs the sample spec asked for, per arm (e.g. 65 for nawawi, "
                        "60 for cayni and mazandarani). Without it the denominator is "
                        "inferred as notes+failures, which UNDERCOUNTS by every bāb "
                        "dropped upstream -- inherited segmentation failures and "
                        "run_naive_batch's sub-100-char skips -- and so overstates "
                        "both yields. Ratios are unaffected; absolute yields are not.")
    p.add_argument("--manifest", default=None,
                   help="the control run's manifest.json. Without it the pairing is "
                        "RECONSTRUCTED, which is only correct for a whole-corpus run "
                        "at the default fraction.")
    args = p.parse_args()

    mf = ROOT / "outputs" / "method_files"
    real_path = (Path(args.real_notes) if args.real_notes
                 else mf / f"{args.commentator}_notes.jsonl")
    control_path = (Path(args.control_notes) if args.control_notes
                    else mf / f"{args.commentator}_control_notes.jsonl")
    real = load_records(real_path)
    control = load_records(control_path)

    if not control:
        print(f"No control records at {control_path} - run scripts/run_control.py first.")
        return 1
    if not real:
        print(f"No study records at {real_path} - the ratio needs both arms.")
        return 1

    if args.no_pairing:
        # The matn-only arm reads its own bāb, so there is nothing to remap. Do NOT
        # let this branch stand in for a missing --manifest on a real control run.
        if args.manifest:
            raise SystemExit("--no-pairing and --manifest are mutually exclusive: "
                             "one says there is no pairing, the other supplies it.")
        pairing = {}
    elif args.manifest:
        pairing = pairing_from_manifest(Path(args.manifest))
    else:
        from hermeneutics.control import shuffle_within_kitab
        pairing = shuffle_within_kitab(load_babs(args.commentator), seed=args.seed)
        print("WARNING: pairing reconstructed, not read from the run. Correct only for "
              "a whole-corpus control at the default fraction; pass --manifest "
              "otherwise, or the verdict below is scored against the wrong texts.")

    # A bāb the model gave up on is the control's expected signal, not a lost sample.
    # _write_failure names these off the notes stem, so the control arm's land in
    # <name>_control_failures.jsonl beside its notes.
    real_failures = len(load_records(failures_for(real_path)))
    control_failures = len(load_records(failures_for(control_path)))
    real_sources, control_sources = commentary_sources(args.commentator, pairing)
    result = score(real, control, real_sources, real_failures, control_failures,
                   control_sources=control_sources, requested=args.requested)

    for arm in ("real", "control"):
        missing = result[f"n_{arm}_unaccounted"]
        if missing < 0:
            raise SystemExit(
                f"--requested {args.requested} is below the {arm} arm's "
                f"notes+failures count; the sample spec and the run disagree.")
        if missing:
            print(f"NOTE: {missing} {arm}-arm bāb(s) requested but never reached "
                  f"(upstream skip or inherited segmentation failure). Counted in the "
                  f"denominator.", file=sys.stderr)

    # The arm has to be in the filename: --no-pairing scores a DIFFERENT arm against the
    # same study notes, and a shared name silently destroys the pre-registered control's
    # scores. Recovering those cost a `git checkout` last time; they are evidence.
    arm = "matnonly" if args.no_pairing else "control"
    out_path = ROOT / "outputs" / "evaluation" / f"{args.commentator}_{arm}_score.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    if result["claim1_withdrawn"]:
        print("\n*** PRE-REGISTERED RULE TRIGGERED: Claim 1 is withdrawn. ***")

    if args.blind_sample:
        sample = build_blind_sample(real, control, seed=args.seed)
        Path(args.blind_sample).write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nBlinded rating file ({len(sample)} records): {args.blind_sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
