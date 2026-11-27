"""Token-aware chunking. Fixes defect #1 (input truncation).

The old pipeline truncated input to 3 000 characters and silently lost 40 % of Nawawi and 61 % of
ʿAynī. Nothing here may truncate: chunk_text returns every word.
"""
import math
import os
import warnings
from functools import lru_cache
from typing import Callable

# Measured over the parsed corpus: 27.71M chars for ~11.1M tokens. Used only for
# planning estimates (count_chunks); real chunking uses the real tokenizer.
CHARS_PER_TOKEN = 2.5

DEFAULT_MAX_TOKENS = 6000  # spec §6.1 recommendation


@lru_cache(maxsize=1)
def _tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


# Conservative: overestimating tokens yields smaller chunks, never oversized ones.
FALLBACK_CHARS_PER_TOKEN = 2.0


def token_len_for(model_id: str) -> Callable[[str], int]:
    """Real token counter for the pinned model.

    Hosted gateways (SAIA) serve model ids that are not HuggingFace repo ids. Set
    VLLM_TOKENIZER_ID to the matching repo to keep exact counts; without it this
    falls back to a conservative character estimate. chunk_text is lossless either
    way — an approximate count shifts chunk boundaries, it never drops words.
    """
    repo = os.environ.get("VLLM_TOKENIZER_ID", model_id)
    try:
        tok = _tokenizer(repo)
    except (OSError, ValueError) as exc:
        warnings.warn(
            f"No tokenizer for {repo!r} ({exc.__class__.__name__}); using a "
            f"{FALLBACK_CHARS_PER_TOKEN} chars/token estimate. Set VLLM_TOKENIZER_ID "
            "to the matching HuggingFace repo for exact counts.",
            RuntimeWarning, stacklevel=2,
        )
        return lambda text: math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def _split_long_word(word: str, max_tokens: int, token_len: Callable[[str], int]) -> list[str]:
    """Split a single word that exceeds max_tokens at character boundaries."""
    # ponytail: binary search for the cut point; only called for words the corpus
    # never actually produces (Arabic prose always has spaces). Add a smarter
    # grapheme-aware split only if real corpus words ever hit this path.
    if token_len(word) <= max_tokens:
        return [word]
    parts: list[str] = []
    while token_len(word) > max_tokens:
        lo, hi, best = 1, len(word), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if token_len(word[:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        parts.append(word[:best])
        word = word[best:]
    if word:
        parts.append(word)
    return parts


def chunk_text(text: str, max_tokens: int, token_len: Callable[[str], int]) -> list[str]:
    """Split on whitespace into chunks of at most max_tokens. Lossless."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        sub_words = _split_long_word(word, max_tokens, token_len)
        for sw in sub_words:
            if current and token_len(" ".join(current + [sw])) > max_tokens:
                chunks.append(" ".join(current))
                current = [sw]
            else:
                current.append(sw)
    if current:
        chunks.append(" ".join(current))
    return chunks


def count_chunks(n_chars: int, max_tokens: int = DEFAULT_MAX_TOKENS) -> int:
    """Planning-only estimate of the chunk fan-out for a unit of n_chars."""
    if n_chars == 0:
        return 0
    return max(1, math.ceil(n_chars / (max_tokens * CHARS_PER_TOKEN)))
