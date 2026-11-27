"""Isnad narrator-chain parsing and graph construction. Spec §3.5, defect #9.

Extracted from notebooks/01_isnad_network.ipynb (T009-T011) rather than reinvented:
the blocklist, the marker regex and the NER/fallback split are prior work, and
re-deriving them from scratch would silently produce different narrator splits
than the ones the notebook's validation checklist was written against.

Node identity: identical name strings collapse to the same graph node. No
disambiguation is attempted here -- that was true of the notebook too.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Kinship terms, transmission markers and honorific/epithet fragments that the
# splitter would otherwise mistake for a narrator name. Verbatim from the notebook.
RELATIONAL_TERM_BLOCKLIST = {
    # Kinship terms
    "أبيه", "أبوه", "أبا", "أبو", "أمه", "أمها", "أخيه", "أخوه",
    "زوجه", "زوجته", "ابنه", "ابنته", "بنته", "جده", "جدته",
    "عمه", "عمته", "خاله", "خالته",
    # Transmission markers
    "حدثنا", "حدثني", "أخبرنا", "أخبرني", "أنبأنا", "أنبأني",
    "عن", "قال", "سمعت", "أخبره", "لـ",
    # Honorific fragments
    "رضي", "عنه", "عنها", "عنهم", "عليه", "عليها",
    "صلى", "وسلم", "تعالى",
    # Prophet epithets
    "النبي", "الرسول", "المصطفى",
}

# Fallback split markers when camel-tools NER is unavailable (no B-PERS/I-PERS tags).
# The notebook listed only the first-person-plural forms. Bukhari's sample carries
# حدثني 140 times and أخبرني 50, and every unsplit occurrence welds two narrators
# into a single node ("بن أبي عدي أنبأنا شعبة" was one node until this).
_MARKERS = re.compile(r"(عن|قال|حدثنا|حدثني|أخبرنا|أخبرني|أنبأنا|أنبأني|سمعت|أخبره|قال لـ)")
# Source markup, stripped before anything else looks at the text. The corpus wraps
# lines with `~~` and carries `msNNNN` manuscript sigla, and neither the NER path nor
# the fallback treats them as anything -- so they rode along INSIDE narrator names. The
# first build of bukhari_metrics.json held `أنس` and `~~أنس رضي الله` as separate nodes:
# 38 nodes carrying latin/digits, 125 carrying honorifics, one man counted twice and his
# betweenness split between the halves. Not a linguistic judgement, so not NER's job.
_MARKUP = re.compile(r"~~|\bms\d+\b|PageV\d+P\d+|@[A-Z]+@|(?:^|(?<=[ (]))\d+\s*-|#")
# `رضي الله عنه` is the written form, but a wrap marker can truncate it to `رضي الله`,
# which is why the honorific survived stripping in the first build.
_RIDWAN = re.compile(r"رضي الله(\s*عنه[ما]?)?")
_SALLA = re.compile(r"صلى الله عليه وسلم")
_ALAYHI = re.compile(r"عليه السلام")
_PARENS = re.compile(r"[()]")
# Arabic comma, semicolon and full stop. `أبيه` is blocklisted; `أبيه،` is a different
# string, so the comma carried a relational term past the check and it surfaced as the
# highest-betweenness node in the Kāfī graph. Punctuation also forked one narrator into
# two nodes (`محمد بن يحيى` and `محمد بن يحيى،`), splitting their centrality.
_PUNCT = re.compile(r"[،؛,.:؛،]+")

_ner_recognizer = None
_ner_available = None


def _ner():
    """Lazy singleton: the arabert NER model is loaded at most once per process.

    Optional path -- environments without the model (this one included; see
    the caught OSError) fall back to the regex+blocklist splitter, exactly as
    the notebook does when NER_AVAILABLE is False.
    """
    global _ner_recognizer, _ner_available
    if _ner_available is None:
        try:
            from camel_tools.ner import NERecognizer
            _ner_recognizer = NERecognizer.pretrained("arabert", use_gpu=False)
            _ner_available = True
        except Exception:
            _ner_recognizer = None
            _ner_available = False
    return _ner_recognizer, _ner_available


def extract_narrators(isnad_text: str) -> list[str]:
    """Split an isnad string into an ordered list of narrator names.

    Two paths, exactly as notebook 01 T009 has them:
    1. camel-tools NER (B-PERS/I-PERS tags), when the arabert model is installed.
    2. Fallback: split on transmission markers, strip honorifics, drop blocklist terms.
    """
    if not isnad_text:
        return []
    isnad_text = _MARKUP.sub(" ", isnad_text)

    ner, available = _ner()
    if available and ner is not None:
        tokens = isnad_text.split()
        if not tokens:
            return []
        try:
            labels = ner.predict_sentence(tokens)
        except Exception:
            labels = ["O"] * len(tokens)

        narrators = []
        current = []
        for token, label in zip(tokens, labels):
            if label == "B-PERS":
                if current:
                    name = " ".join(current).strip()
                    if name not in RELATIONAL_TERM_BLOCKLIST and len(name) > 2:
                        narrators.append(name)
                current = [token]
            elif label == "I-PERS":
                current.append(token)
            else:
                if current:
                    name = " ".join(current).strip()
                    if name not in RELATIONAL_TERM_BLOCKLIST and len(name) > 2:
                        narrators.append(name)
                current = []
        if current:
            name = " ".join(current).strip()
            if name not in RELATIONAL_TERM_BLOCKLIST and len(name) > 2:
                narrators.append(name)
        return narrators

    parts = _MARKERS.split(isnad_text)
    narrators = []
    for part in parts:
        part = part.strip()
        if not part or part in RELATIONAL_TERM_BLOCKLIST:
            continue
        part = _RIDWAN.sub("", part)
        part = _SALLA.sub("", part)
        part = _ALAYHI.sub("", part)
        part = _PARENS.sub("", part)
        part = _PUNCT.sub(" ", part).strip()
        # >3 in the notebook, which silently drops every three-letter name -- and in
        # Bukhari's 500-isnad sample that is عمر (61 standalone occurrences), أنس (58),
        # علي (36) and زيد (31), four of the most frequent Companions in the corpus.
        # They survived the notebook only by accident, carrying an unstripped honorific
        # ("أنس رضي الله") that made them long enough to pass, i.e. as mislabelled nodes.
        # The blocklist is what rejects non-names; length is a crude second line and
        # must not be the thing deciding whether علي is in the graph.
        if len(part) >= 3 and part not in RELATIONAL_TERM_BLOCKLIST:
            narrators.append(part)
    return narrators


def build_isnad_graph(hadiths: list[dict], corpus_name: str) -> nx.DiGraph:
    """Directed weighted graph: nodes are narrators, edges are consecutive
    transmission links in a chain, weight = number of hadiths sharing that edge.

    Hadiths whose isnad yields fewer than two narrators contribute nothing (a
    single name has no transmission link to record) and are silently skipped,
    exactly as notebook 01 T010/T011 does.
    """
    graph = nx.DiGraph()
    edge_counts = defaultdict(int)

    for hadith in hadiths:
        narrators = extract_narrators(hadith.get("isnad", ""))
        if len(narrators) < 2:
            continue
        for narrator in narrators:
            if narrator not in graph:
                graph.add_node(narrator, corpus=corpus_name)
        for i in range(len(narrators) - 1):
            edge_counts[(narrators[i], narrators[i + 1])] += 1

    for (source, target), count in edge_counts.items():
        graph.add_edge(source, target, weight=count)

    return graph


def load_sample(corpus: str) -> list[dict]:
    """Load poc/data/<corpus>_sample.json: the 500-item sample notebook 01 (seed 42)
    drew from the raw corpus and that scripts/run_naive_batch.py etc. already treat
    as the corpus. Building from this rather than re-sampling *_raw.json avoids a
    second, possibly divergent, re-derivation of the same sample.
    """
    path = DATA_DIR / f"{corpus}_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))
