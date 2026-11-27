from hermeneutics.chunking import CHARS_PER_TOKEN, chunk_text, count_chunks


def fake_len(text: str) -> int:
    """Deterministic stand-in for a real tokenizer: 1 token per 4 chars."""
    return max(1, len(text) // 4)


def test_short_text_is_one_chunk():
    assert len(chunk_text("short text", max_tokens=100, token_len=fake_len)) == 1


def test_long_text_splits_into_multiple_chunks():
    text = "ا" * 4000  # ~1000 fake tokens
    assert len(chunk_text(text, max_tokens=100, token_len=fake_len)) > 1


def test_no_chunk_exceeds_the_budget():
    text = " ".join("كلمة" for _ in range(2000))
    for c in chunk_text(text, max_tokens=100, token_len=fake_len):
        assert fake_len(c) <= 100


def test_chunking_loses_no_words():
    """Defect #1 was silent truncation. Every word must survive."""
    text = ("كلمة " * 900).strip()
    chunks = chunk_text(text, max_tokens=50, token_len=fake_len)
    assert " ".join(chunks).split() == text.split()


def test_splits_on_whitespace_not_mid_word():
    words = [f"كلمة{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), max_tokens=50, token_len=fake_len)
    rejoined = " ".join(chunks).split()
    assert rejoined == words


def test_empty_text_yields_no_chunks():
    assert chunk_text("", max_tokens=100, token_len=fake_len) == []


def test_count_chunks_matches_spec_fanout_for_cayni():
    """Spec §6.1 table: ~4,211 commentary calls for ʿAynī at a 6K-token chunk.
    Uses the char heuristic, so allow a band."""
    from scripts.check_corpus import load_babs

    total = sum(count_chunks(len(b["text"]), max_tokens=6000) for b in load_babs("cayni"))
    assert 3800 <= total <= 4650, f"fan-out {total} is outside the spec §6.1 estimate"


def test_chars_per_token_constant_is_documented():
    assert 2.0 <= CHARS_PER_TOKEN <= 3.0
