# Agentic hadith commentary pipeline

The Study-Agent reads commentaries to construct method files; the Query-Agent uses those files and retrieval tools to investigate questions. Each agent is implemented as a LangGraph graph. See RESULTS.md for the study and limitations.

## Installation

With Python 3.11, from the repository root:

```sh
python -m pip install -e .
```

Obtain the source corpora separately. See [data/README.md](data/README.md) and the corpus manifests for sources and expected paths.

The repository contains the `hermeneutics/` package, `scripts/`, `slurm/`, `tests/`, a notebook, source manifests under `data/`, and selected run manifests and scores under `outputs/`.

## Configuration

Set SAIA_API_KEY for GWDG SAIA or OPENROUTER_API_KEY for OpenRouter. Study scripts also use VLLM_BASE_URL, VLLM_API_KEY and VLLM_MODEL_ID; the matched launcher supports STUDY_ENDPOINT, STUDY_API_KEY and STUDY_MODEL. COMPANION_CONTACT optionally supplies the companion contact. See `.env.example` for variable names; keep keys private.

## Reproducing the paper

From the repository root, obtain the data described in `data/README.md`, then run `scripts/run_parse.py` and `scripts/check_corpus.py` for parsing and sample selection. Inspect the remaining stages in order:

```sh
python scripts/run_naive_batch.py --help
python scripts/run_segmentation.py --help
python scripts/run_study_graph.py --help
python scripts/run_control.py --help
python scripts/score_control.py --help
python scripts/run_inquiry.py --help
python scripts/run_question_set.py --help
```

slurm/study_matched.sbatch orchestrates naive, segmentation and study phases. run_control.py supports shuffled and --matn-only controls; score_control.py can use the control manifest to recover the pairing. run_inquiry.py runs the Query-Agent. Consult RESULTS.md and script help for inputs, settings, and corrected denominators. These help commands do not themselves reproduce runs: source corpora, model access, matching settings and substantial compute are required. Hosted model aliases can change.

## Seeds, prompts and manifests

RESULTS.md records the matched sample specifications: Nawawi 14:5:20260914, Cayni 12:5:20260914, Mazandarani 5:12:20260914. scripts/check_corpus.py implements selection. Control shuffling defaults to 20260821 in hermeneutics/control.py and scripts/run_control.py; the control manifest records seed, sample and pairing. scripts/build_isnad_sample.py uses seed 42; scripts/export_network_metrics.py uses 20260821.

hermeneutics/prompts.py defines prompts and their hashes; hermeneutics/manifest.py builds manifests. Saved manifests are in outputs/runs/*/manifest.json. hermeneutics/inquiry_session.py adds Query-Agent fields. Not every manifest records every seed or sample. RESULTS.md records the 2026-09-14 change from grounding rate to yield as primary metric.

## Licences and data

Code is MIT licensed. Third-party texts are not covered by MIT. OpenITI commentary, hadith, lexicon and rijal sources have source-specific terms that must be checked. The Qur'an manifest states Tanzil CC BY 3.0 with verbatim-copy and attribution conditions, and describes separate Quranic Arabic Corpus morphology terms. Obtain sources under their applicable conditions. Author-generated outputs are offered under CC BY 4.0 to the extent the author has rights; embedded source text remains subject to its own conditions.

## Citation

Use the repository's CITATION.cff and add the archived release DOI when available. Cite the accompanying paper separately: Muhammed Ragab, “Reading over the Commentator's Shoulder: How AI Agents Derive and Use the Methods of Sunni and Shīʿī Ḥadīth Commentators,” DH2027, forthcoming.


The repository contains short excerpts from OpenITI digital editions for tests and examples only; see `data/README.md` for attribution and how to obtain the full texts.
