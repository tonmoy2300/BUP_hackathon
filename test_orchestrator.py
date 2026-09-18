from __future__ import annotations

import pytest

from guardrails import GuardrailError
from orchestrator import _check_note_index_coverage, validate_directive_list


# ---------- happy path ----------


def test_two_valid_notes_different_types_pass() -> None:
    raw_list = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [10, 11], "factor": 0.5},
         "explanation": "halve solar"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "off-topic"},
    ]
    result = validate_directive_list(raw_list, note_count=2)
    assert [entry["note_index"] for entry in result] == [0, 1]
    assert result[0]["directive_type"].value == "solar_reduction"
    assert result[1]["directive_type"].value == "no_op"


# ---------- adversarial: shape ----------


def test_not_a_list_rejected() -> None:
    raw = {"note_index": 0, "applies": False, "directive_type": "no_op",
           "structured_adjustment": None, "explanation": ""}
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw, note_count=1)  # type: ignore[arg-type]
    assert exc.value.reason == "directive_list_must_be_list"


def test_too_many_entries_rejected() -> None:
    raw_list = [
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
        {"note_index": 2, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
    ]
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw_list, note_count=2)
    assert exc.value.reason == "directive_list_wrong_length"


def test_too_few_entries_rejected() -> None:
    raw_list = [
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
    ]
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw_list, note_count=2)
    assert exc.value.reason == "directive_list_wrong_length"


# ---------- adversarial: note_index coverage ----------


def test_duplicate_note_index_rejected() -> None:
    """The guardrails-level suite explicitly punts on cross-entry uniqueness;
    the orchestrator is where that gets caught."""
    raw_list = [
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "first"},
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "second"},
    ]
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw_list, note_count=2)
    assert exc.value.reason == "note_index_duplicate"


def test_short_list_with_a_gap_rejected_as_wrong_length() -> None:
    """A list that skips a note_index (0 and 2, for note_count=3) is short by
    one entry, so it's caught by the length gate before the coverage check
    ever runs -- it surfaces as directive_list_wrong_length, not a coverage
    reason. See _check_note_index_coverage's docstring in orchestrator.py:
    given the exact-length gate, a genuine coverage gap can never reach
    validate_directive_list's duplicate check without a duplicate also
    being present (pigeonhole), so note_index_missing is unreachable there
    by construction -- this pins the real, observable behavior instead.
    """
    raw_list = [
        {"note_index": 0, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
        {"note_index": 2, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [5], "factor": 0.3}, "explanation": ""},
    ]
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw_list, note_count=3)
    assert exc.value.reason == "directive_list_wrong_length"


# ---------- white-box: coverage-check helper in isolation ----------


def test_coverage_helper_detects_missing_index_directly() -> None:
    """_check_note_index_coverage is unreachable via validate_directive_list
    (see its docstring), but its own logic is still correct and worth
    pinning directly: called with a genuinely gapped, duplicate-free index
    list -- something the public function's length gate would never let
    through -- it must raise note_index_missing, naming the gap."""
    with pytest.raises(GuardrailError) as exc:
        _check_note_index_coverage([0, 2], note_count=3)
    assert exc.value.reason == "note_index_missing"
    assert "1" in str(exc.value)


def test_coverage_helper_prefers_duplicate_over_missing() -> None:
    """When both conditions hold (as they always do when a gapped list has
    exactly note_count entries), duplicate takes priority as the more
    specific reason."""
    with pytest.raises(GuardrailError) as exc:
        _check_note_index_coverage([0, 2, 2], note_count=3)
    assert exc.value.reason == "note_index_duplicate"


# ---------- adversarial: individual entry propagation ----------


def test_individually_invalid_entry_reason_surfaces_unchanged() -> None:
    raw_list = [
        {"note_index": 0, "applies": True, "directive_type": "not_a_real_directive",
         "structured_adjustment": None, "explanation": ""},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": ""},
    ]
    with pytest.raises(GuardrailError) as exc:
        validate_directive_list(raw_list, note_count=2)
    assert exc.value.reason == "unknown_directive_type"
