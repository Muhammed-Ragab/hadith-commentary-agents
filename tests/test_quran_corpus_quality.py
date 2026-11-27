from pathlib import Path
import json
import re

CORPUS = Path(__file__).parents[1] / "data" / "quran" / "quran_roots.json"

def load_corpus():
    return json.loads(CORPUS.read_text(encoding="utf-8"))

def verse(corpus, sura, aya):
    return next(item for item in corpus if item["sura"] == sura and item["aya"] == aya)

def test_quran_corpus_has_hafs_kufi_counts():
    corpus = load_corpus()
    assert len(corpus) == 6236
    assert len({item["sura"] for item in corpus}) == 114

def test_every_verse_has_text_and_roots_have_no_buckwalter_latin():
    corpus = load_corpus()
    assert all(item["text"] for item in corpus)
    roots = [root for item in corpus for root in item["roots"]]
    assert not any(re.search(r"[A-Za-z]", root) for root in roots)
    assert sum(bool(item["roots"]) for item in corpus) / len(corpus) > 0.5

def test_semantic_spot_checks():
    corpus = load_corpus()
    verse_2183 = verse(corpus, 2, 183)
    assert "كتب" in verse_2183["roots"]
    assert "صوم" in verse_2183["roots"]
    verse_11 = verse(corpus, 1, 1)
    assert verse_11["text"].startswith("بسم الله الرحمن الرحيم")
    verse_91 = verse(corpus, 9, 1)
    assert not verse_91["text"].startswith("بسم الله الرحمن الرحيم")
