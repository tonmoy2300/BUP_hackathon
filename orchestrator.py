from __future__ import annotations

from guardrails import GuardrailError, validate_directive


def _find_duplicate_note_index(note_indices: list[int]) -> int | None:
    """Return the first ``note_index`` value that repeats, or ``None`` if all unique."""
    seen: set[int] = set()
    for idx in note_indices:
        if idx in seen:
            return idx
        seen.add(idx)
    return None


def _check_note_index_coverage(note_indices: list[int], note_count: int) -> None:
    """Verify ``note_indices`` covers ``{0, ..., note_count - 1}`` exactly.

    Runs the duplicate check first, then the missing-index check.

    This helper is deliberately NOT called from ``validate_directive_list``
    below. Given that function's exact-length gate (``len(raw_list) ==
    note_count``) plus each entry's ``note_index`` already being
    range-checked by ``validate_directive``, a call site that reaches this
    logic always has exactly ``note_count`` indices, each in ``range(note_count)``.
    By pigeonhole, N values with no repeats in an N-slot range must be a
    bijection over that range, so a coverage gap can never occur there
    without a duplicate also being present — meaning the missing-index
    branch below is correct but unreachable through
    ``validate_directive_list``. It's kept here, independently tested, in
    case that gate is ever relaxed (e.g. to tolerate short lists) and the
    two conditions become genuinely distinguishable.
    """
    dup = _find_duplicate_note_index(note_indices)
    if dup is not None:
        raise GuardrailError(
            "note_index_duplicate",
            f"'note_index' = {dup} appears more than once in the directive list",
        )

    missing = sorted(set(range(note_count)) - set(note_indices))
    if missing:
        raise GuardrailError(
            "note_index_missing",
            f"'note_index' = {missing[0]} is missing from the directive list",
        )


def validate_directive_list(raw_list: list[dict], note_count: int) -> list[dict]:
    """Validate a full list of LLM-emitted directives against a note batch.

    Each entry is validated individually via :func:`guardrails.validate_directive`
    (any ``GuardrailError`` it raises propagates unchanged). Once every entry
    passes on its own, this checks for duplicate ``note_index`` values.
    Returns the validated dicts sorted by ``note_index``.
    """
    if not isinstance(raw_list, list):
        raise GuardrailError(
            "directive_list_must_be_list",
            f"directive list must be a list, got {type(raw_list).__name__}",
        )

    if len(raw_list) != note_count:
        raise GuardrailError(
            "directive_list_wrong_length",
            f"expected {note_count} directive(s), got {len(raw_list)}",
        )

    validated = [validate_directive(item, note_count) for item in raw_list]

    # Only duplicate detection is reachable here -- see the docstring on
    # _check_note_index_coverage for why a coverage gap (note_index_missing)
    # can't occur independently once the length gate above has passed.
    dup = _find_duplicate_note_index([entry["note_index"] for entry in validated])
    if dup is not None:
        raise GuardrailError(
            "note_index_duplicate",
            f"'note_index' = {dup} appears more than once in the directive list",
        )

    return sorted(validated, key=lambda entry: entry["note_index"])
