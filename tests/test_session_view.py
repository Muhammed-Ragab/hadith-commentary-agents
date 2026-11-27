"""The session viewer's one job that can silently go wrong: pairing each
"[unverified quotation removed]" scar back to the grounding-log entry that caused it.

Get that wrong and the page attributes a removal to the wrong quotation, which is worse
than showing no restoration at all -- it would put words in the agent's mouth on the
slide and in front of the reviewing scholar. The rest of the page is markup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_session_view import SCAR, render, restore_scars


def _log(*entries):
    out = []
    for quote, verified in entries:
        out.append({"quote": quote, "verified": verified, "elided": False, "checked": 1,
                    "in_session": verified, "found_in": ["commentary:x.json"] if verified else [],
                    "foreign_commentary": False, "echoes_question": False})
    return out


def test_scars_pair_in_document_order():
    text = f"before {SCAR} middle {SCAR} after"
    log = _log(("kept", True), ("first bad", False), ("second bad", False))
    out = restore_scars(text, log)
    assert out.index("first bad") < out.index("second bad")
    assert SCAR not in out
    assert "<s>first bad</s>" in out


def test_verified_quotations_are_never_restored_as_removals():
    out = restore_scars(f"a {SCAR} b", _log(("kept", True), ("bad", False)))
    assert "<s>kept</s>" not in out


def test_mismatched_counts_leave_the_gate_text_alone():
    # Two scars, one unverified entry: any pairing would be a guess, so don't guess.
    out = restore_scars(f"{SCAR} and {SCAR}", _log(("bad", False)))
    assert out.count(SCAR) == 2


def test_removal_reason_names_the_advisory_flag():
    log = _log(("echo", False))
    log[0]["echoes_question"] = True
    assert "echoes the question" in restore_scars(f"x {SCAR}", log)


def test_render_is_standalone_and_carries_the_trail_and_the_panel():
    page = render({
        "question": "ما دلالة الحديث؟",
        "condition": "agent_method", "commentator": "cayni", "corpus": "bukhari",
        "manifest_id": "abc123", "method_run": "matched_qwen3_20260917",
        "synthesis_output": f"## راي\n> «{SCAR}»\n",
        "tool_calls_log": [{"tool": "verify_quote", "args": {"quote": "x"},
                            "stage": "synthesis"}],
        "grounding_log": _log(("fabricated", False)),
        "sources": [{"source": "commentary:cayni_babs.json", "reference": "كتاب الإيمان",
                     "excerpt": "..."}],
    })
    assert "<script" not in page and "src=" not in page   # opens from disk, no network
    assert "verify_quote" in page                          # the agentic claim
    assert "<s>fabricated</s>" in page                     # the removal, made visible
    assert 'dir="rtl"' in page                             # Arabic reads right-to-left


def test_an_old_session_without_question_ar_still_renders():
    """A session saved before Part 2 has no question_ar/search_terms keys at all --
    the viewer must not blow up on a KeyError, and must not show a misleading empty
    "Searched as" panel."""
    page = render({
        "question": "ما دلالة الحديث؟", "condition": "no_sources", "commentator": "cayni",
        "corpus": "bukhari", "manifest_id": "abc123", "method_run": "matched_qwen3_20260917",
        "synthesis_output": "answer text", "tool_calls_log": [], "grounding_log": [],
        "sources": [],
    })
    assert "Searched as" not in page


def test_a_new_session_shows_the_canonical_question_and_terms():
    page = render({
        "question": "is it free will or predestination?", "condition": "agent_method",
        "commentator": "cayni", "corpus": "bukhari", "manifest_id": "abc123",
        "method_run": "matched_qwen3_20260917", "synthesis_output": "answer text",
        "tool_calls_log": [], "grounding_log": [], "sources": [],
        "question_ar": "ما حكم الجبر والقدر", "search_terms": ["الجبر", "القدر", "المشيئة"],
    })
    assert "Searched as" in page
    assert "ما حكم الجبر والقدر" in page
    assert "الجبر" in page and "المشيئة" in page


def test_a_rescued_quotation_is_shown_restored_not_struck_through():
    """A session recorded before a gate fix carries that gate's verdicts forever. After
    --reverify, a quotation the old gate stripped and the new one verifies must appear
    as words, not as a fabrication -- it is al-Nawawi's actual text."""
    log = _log(("real after all", False))
    log[0]["stripped_at_run_time"] = True
    log[0]["verified"] = True
    log[0]["found_in"] = ["commentary:nawawi_babs.json"]
    out = restore_scars(f"x {SCAR}", log)
    assert "<s>real after all</s>" not in out
    assert "real after all" in out
    assert "nawawi_babs.json" in out


def test_pairing_follows_the_gate_that_wrote_the_scars():
    """Re-verification flips verdicts but not the synthesis text, so pairing on the NEW
    verdict would count one scar against zero entries and silently render nothing."""
    log = _log(("rescued", True), ("still bad", False))
    log[0]["stripped_at_run_time"] = True     # stripped then, verifies now
    log[1]["stripped_at_run_time"] = True
    out = restore_scars(f"a {SCAR} b {SCAR}", log)
    assert SCAR not in out
    assert "<s>still bad</s>" in out
    assert out.index("rescued") < out.index("still bad")


def test_no_sources_condition_says_so_instead_of_rendering_an_empty_trail():
    page = render({"question": "q", "synthesis_output": "answer",
                   "tool_calls_log": [], "grounding_log": []})
    assert "ran without tools" in page


def test_editorial_corrections_are_shown_not_applied_silently():
    """A corrected session is still the model's output with an edit on top. The page is
    screenshot for a talk and read by a reviewing scholar, so the edit must be visible."""
    page = render({"question": "q", "synthesis_output": "شارح (مازندراني)",
                   "tool_calls_log": [], "grounding_log": [],
                   "corrections": [{"field": "synthesis_output", "from": "مظاندراني",
                                    "to": "مازندراني", "count": 1}]})
    assert "Editorial corrections" in page
    assert "مظاندراني" in page


def test_an_uncorrected_session_carries_no_corrections_note():
    page = render({"question": "q", "synthesis_output": "a",
                   "tool_calls_log": [], "grounding_log": []})
    assert "Editorial corrections" not in page


# --- Part 5: outside_scope, shown next to foreign_commentary -----------------------

def test_a_quotation_flagged_outside_scope_shows_the_badge():
    log = _log(("real but foreign", True))
    log[0]["outside_scope"] = True
    page = render({"question": "q", "commentator": "cayni",
                   "synthesis_output": "answer with «real but foreign»",
                   "tool_calls_log": [], "grounding_log": log})
    assert "outside this run" in page


def test_old_sessions_without_scope_or_outside_scope_still_render():
    """A session recorded before Part 5 has no `scope` key on the session at all, and
    no `outside_scope` key on any grounding_log entry -- the viewer must render exactly
    as it did before, not raise a KeyError or show a stray badge."""
    log = _log(("kept", True), ("bad", False))
    page = render({"question": "q", "condition": "agent_method", "commentator": "cayni",
                   "corpus": "bukhari", "manifest_id": "abc123",
                   "synthesis_output": f"a {SCAR}", "tool_calls_log": [],
                   "grounding_log": log, "sources": []})
    assert "outside this run" not in page
    assert "<s>bad</s>" in page
