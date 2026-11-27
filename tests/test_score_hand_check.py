"""scripts/score_hand_check.py: percent agreement and Cohen's kappa on small synthetic
hand_check.csv/hand_check_key.csv fixtures -- no real hand-scoring needed."""
from scripts.score_hand_check import score


def _write(tmp_path, hand_check_rows, key_rows):
    import csv

    hc = tmp_path / "hand_check.csv"
    with open(hc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "answer_id", "unit_text",
                                          "question", "user_move"])
        w.writeheader()
        w.writerows(hand_check_rows)
    key = tmp_path / "hand_check_key.csv"
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "answer_id", "unit_index", "final_move"])
        w.writeheader()
        w.writerows(key_rows)
    return hc, key


def test_score_perfect_agreement(tmp_path):
    hc, key = _write(
        tmp_path,
        [{"sample_id": "hc_000", "answer_id": "a", "unit_text": "x", "question": "q",
          "user_move": "claim"},
         {"sample_id": "hc_001", "answer_id": "b", "unit_text": "y", "question": "q",
          "user_move": "report"}],
        [{"sample_id": "hc_000", "answer_id": "a", "unit_index": 0, "final_move": "claim"},
         {"sample_id": "hc_001", "answer_id": "b", "unit_index": 1, "final_move": "report"}],
    )

    result = score(hc, key)

    assert result["n_scored"] == 2
    assert result["percent_agreement"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_score_partial_disagreement(tmp_path):
    hc, key = _write(
        tmp_path,
        [{"sample_id": f"hc_{i:03d}", "answer_id": "a", "unit_text": "x", "question": "q",
          "user_move": m}
         for i, m in enumerate(["claim", "claim", "report", "other"])],
        [{"sample_id": f"hc_{i:03d}", "answer_id": "a", "unit_index": i, "final_move": m}
         for i, m in enumerate(["claim", "report", "report", "other"])],
    )

    result = score(hc, key)

    assert result["n_scored"] == 4
    assert result["percent_agreement"] == 0.75


def test_score_skips_unfilled_rows(tmp_path):
    hc, key = _write(
        tmp_path,
        [{"sample_id": "hc_000", "answer_id": "a", "unit_text": "x", "question": "q",
          "user_move": "claim"},
         {"sample_id": "hc_001", "answer_id": "b", "unit_text": "y", "question": "q",
          "user_move": ""}],
        [{"sample_id": "hc_000", "answer_id": "a", "unit_index": 0, "final_move": "claim"},
         {"sample_id": "hc_001", "answer_id": "b", "unit_index": 1, "final_move": "report"}],
    )

    result = score(hc, key)

    assert result["n_scored"] == 1
    assert result["n_unfilled"] == 1
    assert result["percent_agreement"] == 1.0


def test_score_returns_none_when_nothing_is_filled_in(tmp_path):
    hc, key = _write(
        tmp_path,
        [{"sample_id": "hc_000", "answer_id": "a", "unit_text": "x", "question": "q",
          "user_move": ""}],
        [{"sample_id": "hc_000", "answer_id": "a", "unit_index": 0, "final_move": "claim"}],
    )

    result = score(hc, key)

    assert result["n_scored"] == 0
    assert result["n_unfilled"] == 1
    assert result["percent_agreement"] is None
    assert result["cohen_kappa"] is None


def test_score_flags_unmatched_sample_ids(tmp_path):
    hc, key = _write(
        tmp_path,
        [{"sample_id": "hc_999", "answer_id": "a", "unit_text": "x", "question": "q",
          "user_move": "claim"}],
        [{"sample_id": "hc_000", "answer_id": "a", "unit_index": 0, "final_move": "claim"}],
    )

    result = score(hc, key)

    assert result["n_scored"] == 0
    assert result["unmatched_ids"] == ["hc_999"]
