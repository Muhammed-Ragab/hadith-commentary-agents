# Frozen results — matched study, shuffled control, matn-only arm

Status: **final**. Written 2026-09-21, after all four arms were scored and committed.

This file is the citable record of the numbers. It is deliberately not `handoff.md`,
which is untracked (`.gitignore:83`), local to one machine's Drive folder, and rewritten
every working session. `docs/` is ignored too (`.gitignore:65`), which is why this file
lives here. Change it only if a number changes.

Every number below is recomputed from the committed notes files on the **true sample
denominators**. The JSON in `outputs/evaluation/` reports some of them on short
denominators — see "Known defect in the stored JSON".

## Provenance

| | |
|---|---|
| model | `qwen3.8-27b` via SAIA (`chat-ai.academiccloud.de/v1`), temperature 0 |
| study run | `matched_qwen3_20260917` |
| control run | `control_qwen3_20260919` |
| matn-only run | `matnonly_qwen3_20260920` |
| sample specs | nawawi `14:5:20260914` (65 bābs / 14 kitābs), cayni `12:5:20260914` (60 / 12), mazandarani `5:12:20260914` (60 / 5) |
| metric | grounded moves per **requested** bāb; `no_move` and declared failures stay in the denominator |
| pre-registered rule | `control_yield ≥ 0.80 × real_yield` → Claim 1 **withdrawn**, not softened |

The per-arm specs differ, so a single spec for all three would compare different bābs.

## The numbers

Denominators are 65 / 60 / 60 — the requested sample, not the number of notes written.

| arm | condition | notes | `no_move` | moves | **yield** | **ratio to study** |
|---|---|---:|---:|---:|---:|---:|
| nawawi | study | 57 | 4 | 278 | **4.28** | — |
| nawawi | control (shuffled sharh) | 59 | 38 | 95 | **1.46** | **0.342** |
| nawawi | matn-only | 59 | 34 | 47 | **0.72** | **0.169** |
| cayni | study | 57 | 2 | 271 | **4.52** | — |
| cayni | control | 59 | 30 | 115 | **1.92** | **0.424** |
| cayni | matn-only | 59 | 48 | 16 | **0.27** | **0.059** |
| mazandarani | study | 54 | 0 | 292 | **4.87** | — |
| mazandarani | control | 57 | 24 | 157 | **2.62** | **0.538** |
| mazandarani | matn-only | 60 | 48 | 18 | **0.30** | **0.062** |

`real_grounding` and `control_grounding` are **1.0 in every run**.

### Claim 1 — the pre-registered test

**Not withdrawn.** Worst arm is mazandarani at **0.538**, a third below the 0.80 bar.
Mismatched sharh costs the pipeline half to two-thirds of its yield, which is what a real
matn/sharh dependency predicts. `claim1_withdrawn: false` on all three arms, from the
rule as registered, unmodified.

### The matn-only arm — exploratory, NOT pre-registered

Added 2026-09-20, **after** the control numbers were seen. It does not feed the 0.80
rule; that rule is spent and its verdict stands from the control. `claim1_withdrawn` in
the matn-only JSON carries no pre-registration weight. Report as additional analysis,
never as a second falsifier.

**Result: 0.06–0.17 of study yield.** Strip the sharh and the pipeline nearly stops
producing. The `no_move` rate rises to **58–81%** — with only the matn in front of it,
the model mostly declines to claim an interpretive operation at all.

Two things to say explicitly when reporting this:

- **Matn-only scores below the wrong-sharh control, and that is the right direction.** A
  mismatched commentary is still commentary: it has interpretive moves to quote, just the
  wrong ones. A bare matn has almost none. A reviewer expecting no-text ≈ wrong-text will
  otherwise read the gap as an inconsistency.
- **The commentator's name is never given during reading.** `{commentator}` appears in
  exactly one prompt, `_CONSOLIDATE_GLOBAL` (`hermeneutics/nodes.py:920`); the reading
  prompt says only "a classical hadith commentator". A priors channel would have required
  the model to *recognise* al-Nawawī's text, not merely be told whose it was — and it
  produced almost nothing when handed the matn alone.

### Why grounding rate is not the metric

`control_grounding` and `real_grounding` are both 1.0 **by construction**:
`hermeneutics/nodes.py` substitutes the whole cited span and the write gate verifies
it, so an ungrounded move cannot be written in either regime. Grounding rate cannot vary
and therefore cannot falsify anything.

Yield replaced it as the primary metric on **2026-09-14**, threshold unchanged at 0.80.
**This substitution must be disclosed** in any writeup.

## Bāb accounting

Notes + declared failures + inherited segmentation failures = requested, exactly:

| arm | requested | notes | declared failures | segmentation failures |
|---|---:|---:|---:|---:|
| nawawi | 65 | 59 | 2 | 4 |
| cayni | 60 | 59 | 0 | 1 |
| mazandarani | 60 | 60 | 0 | 0 |

(matn-only arm). The segmentation failures are **inherited from the matched study** —
`--segmentation-dir` points at `matched_qwen3_20260917`, which failed 4 nawawi and 1 cayni
bāb. The same bābs are excluded from both arms, so the comparison stays symmetric. A short
notes count is therefore not evidence of a crash.

The two declared nawawi failures are the expected matn-only mode:
`arabic_evidence does not occur verbatim in that evidence span` after two attempts.

## Known defect in the stored JSON

`outputs/evaluation/*_score.json` reports short denominators:
`n_real_requested` reads **60 for nawawi where the spec says 65**, and
`n_control_requested` reads 59 for the cayni control and for the nawawi and cayni
matn-only arms.

**Do not copy yields out of those files.** The affected values are `real_yield`,
`control_yield`, and — where the two arms' denominators differ — `yield_ratio`:

| stored | corrected |
|---|---|
| nawawi `real_yield` 4.6333 | **4.28** |
| nawawi control `control_yield` 1.5833 | **1.46** |
| nawawi matn-only `control_yield` 0.7966 | **0.72** |
| nawawi matn-only `yield_ratio` 0.1719 | **0.169** |
| cayni control `control_yield` 1.9492 | **1.92** |
| cayni control `yield_ratio` 0.4315 | **0.424** |
| cayni matn-only `control_yield` 0.2712 | **0.27** |
| cayni matn-only `yield_ratio` 0.0600 | **0.059** |

No verdict changes: where both arms share a denominator it cancels out of the ratio, and
where it does not, the shift is in the third decimal. The mazandarani arm is unaffected.

The fix belongs in the accounting pass (naive skips; `validate_matched` coverage). Until
it lands, this file is the authority for any published number.

## Regenerating this table

```bash
cd poc
python - <<'PY'
import json, os
SPEC = {'nawawi': 65, 'cayni': 60, 'mazandarani': 60}
ARMS = {'study':   'outputs/method_files_matched/matched_qwen3_20260917/{a}/{a}_notes.jsonl',
        'control': 'outputs/method_files_control/control_qwen3_20260919/{a}/{a}_control_notes.jsonl',
        'matnonly':'outputs/method_files_matnonly/matnonly_qwen3_20260920/{a}/{a}_matnonly_notes.jsonl'}
for a, n_req in SPEC.items():
    base = None
    for label, tpl in ARMS.items():
        p = tpl.format(a=a)
        notes = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        moves = sum(len(d.get('moves') or []) for d in notes)
        nomove = sum(1 for d in notes if not (d.get('moves') or []))
        y = moves / n_req
        if label == 'study':
            base = y
        print(f'{a:12} {label:9} notes {len(notes):3} no_move {nomove:3} '
              f'moves {moves:4} yield {y:.4f} ratio {y/base:.4f}')
PY
```

Any run of this must reproduce the table above exactly. If it does not, something in the
notes files changed and the discrepancy needs explaining before anything is published.

## Caveats to carry into the writeup

- A **no-text arm in the strict sense is untestable** without disabling grounding:
  withholding the sharh outright scores 0.0 by construction, because `build_evidence_catalog`
  emits no spans and every move is rejected at the write gate. That measures the code, not
  the model. The matn-only arm is the closest measurable proxy. State this as a limit.
- Two scoring defects, both fixed, **both failed in the direction "Claim 1 survives"**
  (`d042bf8`, `16c91fc`, `526f90c`). A ratio with yield in the denominator fails toward
  survival whenever either arm's text resolution breaks. Pre-registration protects against
  motivated reasoning, not against a silently wrong instrument. Worth stating plainly.
  `real_grounding: 1.0` is the smoke test — a scoring run showing 0.0 means the arms are
  sharing a source map and the verdict printed under it is meaningless.
- The nawawi matn-only notes file **mixes two generations**: the arm died on a SAIA 429 at
  49/65, and `run_control.py` cannot resume, so the relaunch re-ran the sample from the
  start and appended duplicates. Deduped keeping the first occurrence
  (`nawawi_matnonly_notes.jsonl.bak` retains the raw 108 lines). Same commit, same model,
  temperature 0, so the two generations are equivalent — but say so if anyone asks.
- The predecessor roster is **32 names with perhaps 5 real duplicates left**, cross-script
  (`Mālik`/`مالك`, `al-Khatibi`/`الخطابي`). Normalisation cannot reach those; the real fix
  is upstream. Slated for expert normalisation in the Ḥossam Ouf sitting.

## Query-Agent smoke run

Status: **first real runs**, 2026-09-21. These are smoke tests that the harness works
end to end, **not** a study. Two runs of one question is not a result; no claim in the
paper may rest on the numbers below.

| | |
|---|---|
| model | `qwen3.8-27b` via SAIA (`chat-ai.academiccloud.de/v1`), temperature 0 |
| question | `ما دلالة اختلاف الرواة في حديث النية؟` |
| commentator / corpus | cayni / bukhari (a matched pair) |
| method file | `matched_qwen3_20260917/cayni/cayni_method.md` |

### The runs

| session | condition | sources | tool calls | quotations | verified | rate |
|---|---|---:|---:|---:|---:|---:|
| `bukhari_agent_method_20260921T201359` | agent_method | 129 | 36 | 17 | 10 | 0.588 |
| `bukhari_agent_method_20260921T204047` | agent_method | 102 | 42 | 15 | 8 | 0.533 |
| `bukhari_no_sources_20260921T202802` | no_sources | **0** | **0** | 0 | — |

The second `agent_method` run is after the line-wrap fix below. Session JSON lives in
`outputs/sessions/` and is not committed; the manifest ids are `aab35e9ed732aef4`
and `9b4c472acb6648ac` for the first two.

### What the runs establish

- **The three stages retrieve.** Tool calls were spread across all three
  (isnād 6 / matn 8 / synthesis 22 in the first run). No stage returned empty. Before
  this work the matn stage was instructed to "read the matn content itself" and was
  passed no matn at all.
- **The baseline is clean.** `no_sources` returned zero sources and zero tool calls
  while still answering all three stages. The only difference from `agent_method` is
  access, not capability — which is what makes the comparison legible. A baseline that
  retrieves is a RAG condition wearing the baseline's label.
- **The grounding rate varies, unlike the study arm's.** `real_grounding` is 1.0 in
  every study run *by construction* (see "Why grounding rate is not the metric" above)
  and therefore cannot falsify anything. Here the model writes first and is checked
  after, so 0.533 and 0.588 are measurements. ⚠️ **The conclusion drawn here from that
  spot-check was wrong** — see "The gate was over-censoring" below. The two quotations
  named (`أجمع المسلمون على عظم موقعه`, `مشهور لا متواتر`) genuinely do not occur in
  al-ʿAynī's text, but generalising from two examples to "the gate is not
  over-censoring" did not survive classifying all 26.

### A defect the runs exposed, now fixed

The raw hadith corpora carry `~~`, an OpenITI line-wrap artifact, **mid-word**:
`إنما لكل امرئ ما ~~نوى`. It occurs 33,160 times in `bukhari_raw.json` and 68,067 in
`kafi_raw.json`. Neither normaliser stripped it, so a model quoting the hadith
**correctly** failed verification while one reproducing the artifact passed — the gate
censored honest quotations and rewarded corpus noise.

Fixed in `hermeneutics/grounding.py` and `hermeneutics/corpus_tools.py` by folding `~~`
beside the diacritics and the tatweel.

**This does not touch the frozen numbers above.** The study arms read `data/babs/*.json`,
which contain **zero** occurrences of `~~`. The regeneration script in "Regenerating this
table" was re-run after the fix and reproduced every figure exactly.

Note the direction: unlike `d042bf8`, `526f90c` and `16c91fc`, this defect failed *against*
the pipeline, understating grounding. It is recorded here because a wrong number is a
wrong number whichever way it leans.

### The gate was over-censoring — two more defects, same class, now fixed

All 26 stripped quotations in the six recorded sessions were classified against the
corpus rather than spot-checked. **Ten were the gate's fault, not the model's:**

| cause | n | |
|---|---:|---|
| punctuation only — verbatim in the corpus | 7 | gate defect |
| bāb/kitāb heading — headings were never searched | 2 | gate defect |
| `~~` artifact (the defect above) | 1 | already fixed |
| under 15 chars — a scare-quote, not a citation | 7 | not a citation |
| genuinely not in the corpus | 9 | the model |

**Punctuation.** The OpenITI editions are largely unpunctuated; the model punctuates
what it quotes. `normalize_arabic` folded diacritics, tatweel, alif, tāʾ marbūṭa and
`ﷺ`, but no punctuation — so a quotation identical to the commentary but for one `،`
failed. Every long quote traced broke at its first comma:

```
corpus : لم يصح هذا الحديث عن النبي صلي الله عليه وسلم الا من روايه عمر بن الخطاب ولا عن عمر
model  : لم يصح هذا الحديث عن النبي صلي الله عليه وسلم الا من روايه عمر بن الخطاب، ولا عن عمر
```

**Headings.** `_documents` searched `_record_text`, which returns the body only, while
`_chunks` indexes the reference — so the agent could *find* a bāb heading through
`search_sources`, quote it correctly, and be told it does not occur in the corpus.

Fixed in `hermeneutics/grounding.py` (punctuation folds to a **space**, so the
edition's unspaced `القصد/الوجه` does not glue into a token in neither text) and
`hermeneutics/corpus_tools.py` (the heading becomes its own searchable record, not
concatenated onto the body — a needle spanning that junction would verify against a
string that exists nowhere).

**This does not touch the frozen numbers above.** The yield table is computed from move
counts in the notes files and never calls the normaliser; the regeneration script
reproduces all nine rows exactly. `grounding_rate` *does* call it, and was recomputed
over all nine arms with the fold toggled on and off — **identical to four decimal places
on every arm**. Loosening a substring match can only turn non-matches into matches, and
`real_grounding` was already 1.0.

Re-running the gate over the six recorded sessions rescues 8 of 26 strips and moves
verification from **42/68 (0.618) to 50/68 (0.735)**. This is a change to *measurement*,
not to model behaviour, so it does not invalidate runs made before it.

**One case is knowingly left unfixed.** The edition sometimes glues two words together
(`ولاتحسب` for `ولا تحسب`); a model writing correct Arabic still fails there. Matching
space-insensitively would fix it and was rejected: it makes `عظم موقعه` match
`عظم موقع هذا الحديث`, turning a real contraction into a verified quotation. A gate that
passes a fabrication is worse than one that fails a typo.

Direction, again: this defect ran *against* the pipeline. The earlier note in
"Five runs" generalised from two spot-checks to "the gate is not over-censoring", and
that generalisation was wrong.

### Five runs, and why no rate is quoted from them

The table above reports two runs. Three more followed, after the synthesis stage was
given the sources it had already retrieved (`87e7173` — before that fix the stage was
told to quote and cite while being shown no raw text, the same defect as the original
matn bug one stage later).

| run | synthesis sees sources | sources | tool calls | verify_quote | quotations | verified | rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260921T201359` | no | 129 | 36 | 15 | 17 | 10 | 0.588 |
| `20260921T204047` | no | 102 | 42 | 22 | 15 | 8 | 0.533 |
| `20260921T213404` | yes | 64 | 19 | 8 | 10 | 2 | **0.200** |
| `20260921T215347` | yes | 76 | 40 | 26 | 15 | 12 | **0.800** |
| `20260921T221003` | yes | 80 | 31 | 17 | 11 | 10 | **0.909** |

**The finding is the spread, not the means.** Identical question, identical condition,
identical model, temperature 0: the rate ranges 0.200 to 0.909. Blind-synthesis mean is
0.561 (n=2), sighted-synthesis mean 0.636 (n=3), and that gap is meaningless beside a
0.71 range within one arm.

This is the same result `08feea5` recorded for the earlier comparison work, where the
run-to-run range was wider than the gap between conditions. It is the difference between
a repeated direction and a measurement.

**Re-verified under the fixed gate (`30178ff`, 2026-09-22).** The rates above were written
by a gate that stripped punctuated and heading quotations (see "The gate was
over-censoring"). Re-checking the same recorded quotations, model output untouched:

| run | before | after |
|---|---:|---:|
| `20260921T201359` | 10/17 · 0.588 | 11/17 · 0.647 |
| `20260921T204047` | 8/15 · 0.533 | 10/15 · 0.667 |
| `20260921T213404` | 2/10 · 0.200 | 5/10 · 0.500 |
| `20260921T215347` | 12/15 · 0.800 | 13/15 · 0.867 |
| `20260921T221003` | 10/11 · 0.909 | 11/11 · 1.000 |

The range narrows from 0.71 to **0.50 (0.500–1.000)**. The conclusion does not change: a
0.50 spread across identical runs is still wider than any difference between conditions,
and still no rate may be quoted.

**These five ran at `max_tokens=2000`.** qwen3's reasoning comes out of the same budget,
and calls ending at exactly 2000 with `finish_reason=length` occur in the traces of these
runs. Those calls survived because they also carried tool calls, so the loop continued
on a truncated reply. The traces append across runs sharing a directory, so the
truncation cannot be attributed to individual sessions. It is one more reason these runs
are smoke, not data. The budget is now a flag, default 4000, recorded in the manifest
(`9a08c33`).

**Consequences, which must survive into any writeup:**

- **No grounding rate may be quoted from a single run.** Not in a paper, not in a slide,
  not in the scholar's review sheet as though it characterised the system.
- **The synthesis-sources change was kept, but not because these runs justify it.** It
  was kept because a stage instructed to quote and cite must be able to see what it is
  quoting; that is an argument about the design, not about 0.636 versus 0.561. Reverting
  it on the strength of the 0.200 run would have been the same error in the other
  direction.
- An intermediate hypothesis — that the tool round-trip is what produces verbatim
  fidelity, because a model reading sources from its prompt paraphrases them — was
  raised by the 0.200 run and is **not supported**: the 0.909 run made fewer
  `verify_quote` calls (17) than the 0.533 run (22). Recorded as an open question, not a
  result.
- Any real study of this needs repeats per cell with the spread reported, exactly as the
  study arms above did.

### Known limits of the gate

- **Mis-attribution is now recorded, not prevented.** `verify_quotation` still asks
  whether a quotation occurs anywhere in the corpus — that has not changed, and
  `verified` still means exactly what it meant before. What changed: `verify_quote`'s
  provenance (`occurrences`), previously discarded, now comes back as `sources` on
  `verify_quotation`'s result, computed as the INTERSECTION across every checkable
  fragment of an elided quotation. `_synthesis_gate` uses it to add `found_in` and
  `foreign_commentary` to each `grounding_log` entry: `foreign_commentary` is true when
  a verified quotation was found only in a RIVAL commentator's commentary file (the
  lexica, Qur'an, rijal works and hadith corpus are excluded — quoting those is
  legitimate). The gate does **not** strip on `foreign_commentary`: the text is real, so
  stripping would be wrong. The case this closes — `الفعال الصالح` verifying while
  absent from al-ʿAynī — would now show `foreign_commentary: true` on the review sheet
  instead of silently passing as an ordinary verified quotation. This still cannot check
  a quotation against whatever the model's PROSE claimed as its source; it can only
  compare what the model quoted against where that text actually occurs.
- **Question-echo false positives are now flagged, not silently counted as failures.**
  Both smoke runs quoted fragments of the question itself (`اختلاف الرواة`,
  `اختلافًا بين الرواة`) in quotation marks. `_synthesis_gate` now adds
  `echoes_question` to each `grounding_log` entry — true when the quotation's
  normalised text occurs in the normalised question — and keeps logging the entry
  either way rather than dropping it, so both the raw rate and a rate corrected for
  question-echoes stay computable from the log. The gate does not compute or store a
  rate itself; that stays downstream analysis, and no rate in this file has been
  recomputed.
- **Temperature 0 is not determinism, and this is the binding limit.** Five runs of the
  identical question, condition and model produced rates from **0.200 to 0.909** — see
  "Five runs, and why no rate is quoted from them" above. The spread is wider than any
  difference between the configurations tested, exactly as `08feea5` found for the
  earlier comparison work. Until a cell is run repeatedly and reported with its spread,
  **no grounding rate from this pipeline characterises anything** and none may be
  quoted. This limit outranks the two above: both of them refine *what* is counted,
  while this one says the count itself is not yet a measurement.

## Two-tradition worked example (2026-09-22)

One identical question through both traditions, as the abstract claims ("how that
derived method differs across Sunni and Shia traditions"). **A worked example, n = 1 per
arm, not a measurement** — the five-run spread above applies to it in full.

| | |
|---|---|
| question | «من هم ورثة علم النبي صلى الله عليه وسلم، وما الذي تمنحه هذه الوراثة من سلطة في فهم الحديث؟» |
| condition | `agent_method`, method run `matched_qwen3_20260917` |
| model | `qwen3.8-27b` via SAIA, temperature 0, `max_tokens` 4000 |
| gates | all three `auto_approve` — no human reviewed either run |
| driver | `scripts/run_two_traditions.py` |

The question was chosen against the corpus: both commentaries treat it in their Kitāb
al-ʿIlm at comparable length (al-ʿAynī باب العلم قبل القول والعمل + باب فضل العلم;
al-Māzandarānī باب صفة العلم وفضله + باب ثواب العالم والمتعلم). Obedience to *ulū
l-amr* counted higher and was rejected: al-Māzandarānī's hits fall inside undivided
mega-units (al-Rawḍa is one 1.69M-character record), which are not focused treatments.

| | al-ʿAynī (Bukhārī) | al-Māzandarānī (al-Kāfī) |
|---|---:|---:|
| session | `bukhari_agent_method_20260922T152247` | `kafi_agent_method_20260922T153431` |
| tool calls | 40 | 41 |
| isnād / matn / synthesis | **19 / 3** / 18 | 13 / **16** / 12 |
| distinct tools | 4 (no `lookup_isnad_network`) | 5 |
| sources retrieved | 179 | 78 |
| quotations / verified | 16 / 8 | 26 / 14 |
| calls at the token ceiling | 2 | 0 (peak 1,959) |
| tool-round cap reached | no | no |

> ⚠️ **Correction, 2026-09-24.** This pair ran before retrieval was scoped to the
> commentator's tradition (`9a2cae6`). The al-ʿAynī arm read **33 al-Kāfī and 15
> al-Māzandarānī passages** of ~170, so its isnād-heavy split was measured while partly
> reading Shia sources. None of its final quotations came from a rival commentary. The
> scoped re-run (question q7 in "Conference question set" below) keeps the direction
> but not the magnitude: al-ʿAynī 9 / 6 / 6, al-Māzandarānī 19 / **30** / 13. Treat the
> table below as superseded by that one.

**What it shows.** The allocation of effort inverts: al-ʿAynī's run spends its tool calls
on the isnād, al-Māzandarānī's on the matn. That is the direction the operation profile
predicts independently (al-ʿAynī: `narration_harmonization` 26,
`counterargument_refutation` 24; al-Māzandarānī: `tawil_imposition` 34,
`sense_bifurcation` 38). Two measurements agreeing is the claim; either alone is not.

**What it does not show.**
- Nothing about stability. One run per arm; the split may not survive repeats.
- The verification rates (0.50, 0.54) differ by less than the run-to-run spread. Not a
  result.
- The al-ʿAynī arm hit the 4000-token ceiling twice. Both calls survived because they
  carried tool calls, but the ceiling still binds on that arm and not on the other.
- Neither arm reached the 8-round tool cap, so the isnād count is the model's own
  allocation, not a cap artifact.

**One editorial correction.** The al-Māzandarānī session wrote the commentator's name
«مظاندراني» (ظ for ز) four times, transliterating the Latin filename, since no prompt
gave it the Arabic name. Corrected to مازندراني in the session file and recorded in its
`corrections` list; none of the four was inside a quotation, so no verdict moves. The
raw trace keeps the original. Runs after `4bb7d6e` receive the Arabic name in every
search hit, so **they are not strictly comparable with this pair**. The change is to
tool output, not to a prompt template, so `prompt_hashes` does not capture it.

This does **not** touch the study arms' guarantee above ("the commentator's name is never
given during reading"). The `author` field lives in `corpus_tools.search`, which only the
Query-Agent calls; `nodes.py`, `study_graph.py`, `run_study_graph.py`, `run_control.py` and
`run_naive_batch.py` do not import it. And the Query-Agent was never blind to identity:
every hit already named its file, `mazandarani_babs.json`. The field adds the correct
spelling, not new information.

## Conference question set (2026-09-24)

Worked examples, **one run per cell**. The five-run spread above applies to every row:
nothing here is a rate, and no verification count may go on a slide as though it
characterised the system.

| | |
|---|---|
| Query-Agent model | `qwen/qwen3.8-27b` via **OpenRouter**, pinned to DeepInfra bf16, `allow_fallbacks: false`, temperature 0, `max_tokens` 4000 |
| frontier baseline | `openai/gpt-5.6-luna` via OpenRouter, cheapest upstream (`sort: price`) |
| code | `9a2cae6` (tradition scoping) on branch `query-agent-ui` |
| gates | all `auto_approve`. No human reviewed any run |
| driver | `scripts/run_question_set.py`; index `outputs/question_set/index.json` |
| cost | ≈ $0.095 per Query-Agent run; the whole Luna set, cents |

**Backend change, to disclose.** The frozen study and the two 09-22 pairs ran on SAIA;
this set ran on OpenRouter/DeepInfra. Same model id, different serving stack. The
manifest records `provider`, `provider_route` and `quantization` per run.

### Three pipeline changes made before this set (all change model input)

1. **Canonicalisation** (`18bb495`). Every question, in any language or register, is first
   rewritten into one classical-Arabic question plus Arabic search terms, and the stages
   run on that. Before this, an English question retrieved nothing: search is literal
   term-matching over Arabic text. Recorded as `question_ar`, `search_terms` and
   `question_lang` in each session.
2. **Tradition scoping** (`9a2cae6`). ⚠️ **The defect it fixes affected every Query-Agent
   run before it, including the 09-22 pair.** The tools accepted any `source`, so runs
   drifted across traditions: an unscoped al-ʿAynī run on ra'y/qiyās read al-Kāfī 71,
   al-Māzandarānī 35 and al-ʿAynī 33 passages, and answered with the Imāmī position. Two
   of four unscoped al-Māzandarānī runs read **zero** passages of his own commentary. Now
   each run can search only its own commentary, hadith collection, rijāl and isnād
   metrics, plus the Qurʾān and the lexica, and the matn stage always fetches the
   commentator's own text. Verification stays whole-corpus; a verified quotation found
   only outside the run's scope is flagged `outside_scope`. The unscoped batch is kept in
   `outputs/question_set_v1_unscoped/` and is not used.
3. **Network statistics are not doctrine.** Unscoped answers argued from isnād centrality
   to legal positions ("the ban is one rule from the central node"). The isnād and
   synthesis prompts now forbid it. This lives in `inquiry_graph.py`, not `prompts.py`,
   so `prompt_hashes` does **not** capture it; the manifest's `git_sha` does.

### The runs

Passages *read* (retrieved with text; `verify_quote` hits are excluded). **Rival-tradition
passages read: 0 in all 18 runs.**

| cell | own commentary / own hadith read | quotations verified | tool calls isnād / matn / synthesis |
|---|---|---:|---|
| q1 ra'y & qiyās · ʿAynī | 12 / 12 | 11 of 14 | 12 / 7 / 9 |
| q1 · Māzandarānī | 20 / 96 | 25 of 26 | 19 / 20 / 18 |
| q1 (English) · Māzandarānī | 26 / 68 | 14 of 15 | 22 / 19 / 28 |
| q2 conflicting hadith · Māzandarānī | 62 / 38 | 20 of 26 | 20 / 15 / 12 |
| q3 riwāya bi-l-maʿnā · Nawawī | 56 / 8 | 6 of 7 | 15 / 13 / 9 |
| q3 · Māzandarānī | 20 / 51 | 6 of 6 | 12 / 7 / 3 |
| q4 jabr & qadar · Nawawī | 64 / 43 | 16 of 20 | 19 / 23 / 12 |
| q4 · Māzandarānī | 12 / 63 | 15 of 23 | 16 / 6 / 16 |
| q5 taqiyya · ʿAynī | 26 / 82 | 8 of 11 | 21 / 8 / 7 |
| q5 · Māzandarānī | 8 / 88 | 10 of 15 | 16 / 7 / 6 |
| q5 (English) · Māzandarānī | 18 / 156 | 14 of 15 | 18 / 23 / 9 |
| q5 (colloquial) · Māzandarānī | 20 / 104 | 11 of 13 | 15 / 14 / 8 |
| q6 duʿāʾ · ʿAynī | 18 / 26 | 7 of 7 | 12 / 6 / 8 |
| q6 · Māzandarānī | 34 / 99 | 22 of 29 | 20 / 17 / 14 |
| q6 (English) · Māzandarānī | 26 / 123 | 13 of 13 | 24 / 20 / 13 |
| q7 heirs of the Prophet's knowledge · ʿAynī | 32 / 15 | 14 of 17 | 9 / 6 / 6 |
| q7 · Māzandarānī | 30 / 58 | 23 of 33 | 19 / 30 / 13 |
| q8 «اطلبوا العلم ولو بالصين» · ʿAynī | 154 / 68 | 8 of 9 | 17 / 13 / 11 |

q8's three `outside_scope` flags are the question's own phrase and a kitāb heading, both
found only in al-Māzandarānī's file. They are not borrowed evidence.

**What the pairs show.** On q1 the two readings diverge the way the traditions do, each
quoted from its own commentary. Al-ʿAynī (باب ما يذكر من ذم الرأي وتكلف القياس): ra'y
resting on an aṣl «فهو محمود وهو الاجتهاد», and qiyās «مأمور به فكان حجة». Al-Māzandarānī:
«إن دين الله لا يصاب بالقياس», «من نظر برأيه هلك». Before scoping, both arms gave the
Imāmī answer.

**What they do not show.** Anything about stability (n = 1), and anything from the
verification counts. The q7 stage split keeps the 09-22 direction (ʿAynī isnād > matn,
Māzandarānī matn > isnād) at a much smaller magnitude. Note it at most; do not claim it.

### Invariance: q5 in three registers, al-Māzandarānī

| input | canonical question | search terms |
|---|---|---|
| fuṣḥā | هل يجوز للمسلم أن يتقي بدينه أو يخفي رأيه إذا خاف على نفسه؟ | التقية، الكتمان، الإخفاء، الخوف، الضرورة، الحرج، الجواز |
| English | هل يجوز للمسلم أن يخفي دينه أو رأيه إذا خاف على نفسه؟ | التقية، الكتمان، الإكراه، الخوف على النفس، إظهار الكفر، العذر، المخافة |
| Egyptian colloquial | هل يجوز كتمان الدين أو الرأي عند خوف على النفس من الناس؟ | التقية، كتمان الدين، الخوف على النفس، الإخفاء، الضرورة، العذر الشرعي |

The **question** converges: three near-identical canonical questions, all anchored on
التقية and الكتمان. The **retrieval** overlaps only partly: Jaccard over passages read,
0.41 (fuṣḥā–English), 0.43 (fuṣḥā–colloquial), 0.31 (English–colloquial). But all three
answers carry the **same two verified key quotations**, «تسعة أعشار الدين في التقية» and
«لا دين لمن لا تقية له». The evidence the reading rests on converges even where the
wider retrieval does not. With one run per cell, the language effect cannot be separated
from run-to-run spread: claim the convergence of question and core evidence, not
identical retrieval.

### Frontier baseline: one plain call, no pipeline

`scripts/run_frontier_baseline.py`. System prompt `conditions.BASE` only, the bare
question, one call. Quotations are checked against the **whole** corpus by the same
`verify_quotation`, and nothing is stripped. Output
`outputs/frontier/frontier_openai_gpt-5.6-luna_20260924T161907/`.

An earlier Luna batch through `run_inquiry --condition no_sources`
(`kafi_no_sources_20260924T1607*`–`1609*`) is **not** a plain baseline: `no_sources`
still runs the three-stage graph and hands the model isnād metrics and a corpus label,
and the answers comment on "the data shown". Superseded; do not use.

| | |
|---|---|
| answers | 11 (8 Arabic, 3 English) |
| quotations | **9 in total; 7 of the 11 answers quote nothing** |
| verified | 7 of 9 |
| output tokens | 384–1,249 per answer |
| answers naming any commentator | **0** |

**The expected finding did not appear.** The run plan (§5) anticipated a frontier model
confidently fabricating Arabic quotations. Luna did not. Both unverified quotations are
defensible: «اطلبوا العلم ولو في الصين» is the phrase the question asks about (which Luna
correctly calls very weak or fabricated), and «العلماء ورثة الأنبياء…» is a real hadith
attributed to Abū Dāwūd and al-Tirmidhī, which are outside this corpus. **"Unverified"
here means "not in this corpus", not "fabricated".**

**The difference that does appear is voice, not honesty.** Luna gives one careful,
generic answer, from the Sunni side by default. On q5 it cites Q 16:106 and ʿAmmār; the
al-Māzandarānī run builds on «تسعة أعشار الدين في التقية» and «لا دين لمن لا تقية له».
It names no commentator and cannot say how al-ʿAynī and al-Māzandarānī each read a
question. State the comparison that way. "The chatbot fabricates" is not supported by
this data.

### Level 2: mushkil al-ḥadīth under an outside objection (2026-09-27)

The talk presents the set in two levels: **level 1** (q1–q8 above) asks a tradition
about its own questions; **level 2** asks it to answer an objection from outside.
Suggested informally by Ḥossam Ouf: a well-known problematic ḥadīth, an objection voiced
from European ethics, and a request to gather the narrations, grade them, link the
authentication to the reading, and offer a *Deutungsangebot* that keeps the classical
commentators. q9 al-šuʾm, q10 the prayer ḥadīth (plus English), q11 umirtu. The questions
deliberately do not hint at variant wordings or at ʿĀʾiša's objection. Same model, driver
and routing as above; the only code change is the new questions in `run_question_set.py`. One run per cell, all `auto_approve`.

Key variant: **R** = retrieved, **A** = used in the answer, **–** = not found (normalised
substring match, `scratchpad` script, not committed).

| cell | quotations verified | key variant(s) found |
|---|---:|---|
| q9 al-šuʾm · ʿAynī | 28 of 30 | conditional *in kāna l-šuʾm* R A |
| q9 · Māzandarānī | 13 of 18 | *in kāna* R A, **from Lisān, not al-Kāfī**; states it cannot grade the ḥadīth from its sources |
| q10 prayer · ʿAynī | 23 of 29 | ʿĀʾiša's objection R A; *lā yaqṭaʿu… šayʾ* R A; naskh discussed |
| q10 · Nawawī | 11 of 16 | naskh R A; **ʿĀʾiša missed**, though it is in Muslim |
| q10 · Māzandarānī | 18 of 20 | al-Kāfī's *lā yaqṭaʿu l-ṣalāta šayʾ, lā kalb wa-lā ḥimār wa-lā mraʾa* R A |
| q10 (English) · Māzandarānī | 3 of 3 | none; thin answer |
| q11 umirtu · ʿAynī | 23 of 27 | *al-nās* read as the Arab mušrikūn R A; jizya R A |
| q11 · Nawawī | 10 of 11 | *al-nās* as the mušrikūn R A |

`outside_scope` = 0 in all eight.

**What works.** On q10 each tradition answers from its own material. Al-ʿAynī sets
ʿĀʾiša's «شبهتمونا بالحمر والكلاب» against Abū Ḏarr, glosses *qaṭʿ* as loss of
concentration («المراد من القطع: النقص، لشغل القلب»), and reports «وهو قول عامة العلماء».
Al-Māzandarānī answers with al-Kāfī's counter-narration, and his answer concedes the
objection is well-founded on a literal reading («وهذا الإشكال مؤسَّس إذا قُرئ الحديث
حرفيًّا») before answering it. **Retrieval did not discover the counter-evidence; the
model already knew it.** The ʿAynī run's 4th search already contains «عائشة» and its 7th
call verifies «شبهتمونا بالحمر والكلاب»; the Māzandarānī run searches «لا يقطع صلاة المسلم
شيء» by name. What the pipeline adds is that each recalled quotation is then checked
against the corpus and kept inside one tradition. The q9 al-Māzandarānī run
declined to invent a grading: «لا أستطيع… بيان درجته… من هذه المصادر».

**What does not.**
- **Retrieval misses the best passage twice.** Al-Māzandarānī says the Prophet said
  al-šuʾm «حكاية عن غيره» (`mazandarani_babs.json`); the q9 run never retrieved it. The
  Nawawī q10 run missed ʿĀʾiša's objection, which is in `muslim_raw.json`.
- **The interpretive proposal defends more than it engages.** q10 ʿAynī closes by calling
  the objection a reading that «لا تستقيم», and adds an argument of its own that is in no
  source (Bukhārī would not have included ʿĀʾiša's ḥadīth if the likening were meant).
  "Takes the modern objection seriously" came out as rebuttal.
- **Network numbers again.** q10 ʿAynī prints a centrality table and says the "cutting"
  narration has an almost single path while ʿĀʾiša's has many. It disclaims the inference
  in the next sentence, but this is the move the isnād prompt forbids.
- Bukhārī numbers in the answers (e.g. «البخاري ٤٩٢») are not checked against a standard
  numbering.
- The English q10 run is much weaker than the Arabic one (3 quotations, no counter-narration).

**Frontier baseline (Luna, no sources).**
`outputs/frontier/frontier_openai_gpt-5.6-luna_20260927T141345/`. Unlike on q1–q8, Luna
quotes heavily here: q9 9 of 19 verified, q10 8 of 16, q11 12 of 17, English q10 none.
**Superseded:** all four answers in that run stopped at `max_tokens=4000` mid-sentence.
Rerun at 16,000 (`frontier_openai_gpt-5.6-luna_20260927T151354/`, each ends naturally at
3.7k–4.4k tokens): q9 12 of 19, q10 4 of 12, q11 6 of 12, English q10 1 of 3. The same
question twice moved q11 from 12 to 6 verified; single Luna runs are noisy.
The unverified ones are again **not fabrications**: vocalised or slightly variant wordings
of real narrations (e.g. «بئسما عدلتمونا بالكلاب والحمير» against the corpus's «بالكلب
والحمار»), narrations outside this corpus (ʿĀʾiša's «قاتل الله اليهود…» on al-šuʾm), and
Luna's own paraphrases of the objection set in quotation marks. Luna knows the classical
dossier from memory, including ʿĀʾiša's objection to al-šuʾm, which neither agent run found.

**What to claim.** On these famous cases the agent does not out-know a frontier model.
What it adds is an answer scoped to one tradition, with every quotation traceable to a
passage, and a stated limit where its sources stop. It does not add better coverage.

### Agent vs Luna across both levels: measured comparison (2026-09-27)

`scripts/compare_systems.py` scores every final answer of both systems with the same
code: 26 agent answers (Qwen3.8-27B, one per question × commentator) and 15 Luna answers
(one plain call per question; level 2 from the 16k rerun). Output
`outputs/comparison/20260927T173044Z/` (`metrics.csv`, `similarity.csv`, `keyness.csv`,
`entities.csv`, `review/<qid>.md` side-by-side sheets, `README.md`). Lexicons and the
level-2 key-variant list are editable in `scripts/compare_lexicons.json`. n = 1 per cell:
descriptive only, no significance tests. Luna gives one answer per question, the agent one
per commentator.

Medians (agent L1 n=18, L2 n=8; Luna L1 n=11, L2 n=4):

| | agent L1 | agent L2 | Luna L1 | Luna L2 |
|---|---:|---:|---:|---:|
| words | 655 | 1,031 | 220 | 1,493 |
| quotations | 15 | 21 | 0 | 12 |
| verified rate | 0.84 | 0.81 | (0.88)* | **0.42** |
| commentators named | 3.5 | 7.5 | 0 | 1 |
| attributed-sentence share | **0.44** | 0.38 | 0.12 | 0.23 |
| hedges per 1k words | 3.5 | 3.9 | **15.1** | 5.3 |
| stated limits per answer | 0 | **2.5** | 0 | 0 |
| grading terms | 1 | 8 | 1 | 10.5 |

\* 7 of Luna's 11 level-1 answers quote nothing; pooled it is 7 of 9. Pooled, the agent
verifies 243 of 299 (L1) and 138 of 173 (L2); Luna 23 of 46 at L2.

**Verification holds for the agent and halves for Luna at level 2.** Hamza spelling does
not explain Luna's drop: `normalize_arabic` keeps «ء» but folds «ئ» (so «شيء» ≠ «شئ»);
re-verifying every unverified quotation with «ء» folded changes one agent quotation and no
Luna quotation. "Unverified" still means "not in this corpus", not "fabricated" (see above).

**Voice.** The agent attributes (قال، روى، أخرجه, or a quotation) in 44% of sentences
against Luna's 12% at level 1, names the commentators, and at level 2 says where its
sources stop. Luna answers in its own voice and hedges four times as often.

**Text reuse is the clearest tradition signal.** Share of each answer's word 4-grams found
verbatim in each commentary (median):

| answer | in al-ʿAynī | in al-Nawawī | in al-Māzandarānī |
|---|---:|---:|---:|
| agent, ʿAynī arm | **0.110** | 0.019 | 0.010 |
| agent, Nawawī arm | 0.057 | **0.107** | 0.009 |
| agent, Māzandarānī arm | 0.006 | 0.000 | **0.088** |
| Luna | 0.037 | 0.022 | 0.011 |

Each arm carries its own commentator's wording about ten times more than the others'.
Honorific formulas agree: Māzandarānī arm imāmī-style 1 / sunnī-style 0 (medians), ʿAynī
and Nawawī arms 0 / 2, Luna 0 / 2. Luna's default is Sunni.

**Embeddings separate systems, not traditions.** Silhouette (cosine,
`paraphrase-multilingual-MiniLM-L12-v2`, Arabic answers, n=33): by system 0.24, by
tradition −0.04, by question −0.11. Controls: markdown stripped and ≤8 × 150-word chunks,
by system 0.13; level 2 only (similar lengths) 0.36; tradition within the agent only
−0.02; length terciles −0.05; Spearman(word count, PC1) −0.22, p = 0.23. The system
difference is not length. The traditions differ in wording and evidence (reuse,
honorifics, quotations), not in what a general-purpose embedding sees as topic.

**Level-2 key variants** (answer contains it, normalised; agent also: retrieved). ʿAynī
0.75–1.0, Nawawī 0.6–0.75, Luna 0.4–1.0 (English q10 0), Māzandarānī 0–0.2. The list was
Sunni-weighted; Imāmī wordings were added for q10 (a fragment rule now catches «لا يقطع
صلاةَ المسلم شئٌ») and for q9 al-Kāfī 9815: «فأما شؤم المرأة فكثرة مهرها وعقم رحمها».
Not chosen as *the* Imāmī answer; it is in the corpus, so an answer from that tradition
should be able to find it. The same gloss is in al-ʿAynī («عقرها وغلاء مهرها»), so the
item counts either wording: the reading of al-šuʾm as a circumstance (barrenness, a high
dowry), not womanhood, is shared by both traditions. The ʿAynī q9 arm uses it; the
**Māzandarānī q9 arm never retrieved 9815**, its second miss in its own corpus after
«حكاية عن غيره». Its q9 zero is a retrieval failure, not a gap in the list. Luna recalls
the dossier about as well as the Sunni arms; what differs is verification.

**Dropped: argument moves.** A rule tagger plus a Haiku 4.5 pass
(`scripts/tag_moves_llm.py`, $0.17, prompt hash `453828313f2eb29f`) labelled answer units
as claim / report / quotation / objection / reply / limit / proposal. Not used: units split
at every list item (median 13 words in the hand-check sample, some a bare citation line),
too small to judge, so the labels could not be validated. The `move_*` columns in
`metrics.csv` and the move-strip figure are unvalidated; do not cite them. The label-free
counterparts above (attribution share, quotation paragraphs 30% vs 15%, stated limits)
carry the point. Redo with paragraph-sized units if the paper needs it.

**What to claim.** Same as level 2 above, now measured: the agent does not know more, it
answers from one tradition's own words (reuse, honorifics, attribution), keeps its
quotations verifiable when the question gets hard, and states where its sources stop.
Luna gives one hedged, Sunni-leaning answer in its own voice.
