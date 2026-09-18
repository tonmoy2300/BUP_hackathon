from __future__ import annotations

import math
from enum import Enum
from typing import Any


class _FallbackDirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


try:  # Prefer the canonical enum from schemas.py when available.
    from schemas import DirectiveType as _CanonicalDirectiveType

    if all(getattr(_CanonicalDirectiveType, m.name, None) for m in _FallbackDirectiveType):
        DirectiveType = _CanonicalDirectiveType
    else:
        DirectiveType = _FallbackDirectiveType
except Exception:  # pragma: no cover - keeps guardrails usable in isolation
    DirectiveType = _FallbackDirectiveType


class GuardrailError(ValueError):
    """Raised when an LLM-emitted directive dict fails validation.

    ``reason`` is a short, stable identifier suitable for logging and tests.
    The full message includes context for humans.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"[{reason}] {message}")
        self.reason = reason


# Required keys for the structured_adjustment payload of each non-no_op type.
# Anything outside this set is permitted (LLMs often add explanatory fields),
# but every key listed here must be present.
_REQUIRED_ADJUSTMENT_KEYS: dict[DirectiveType, tuple[str, ...]] = {
    DirectiveType.SOLAR_REDUCTION: ("hours", "factor"),
    DirectiveType.MINIMUM_BATTERY_RESERVE: ("hours", "minimum_energy_kwh"),
    DirectiveType.NO_CHARGE_WINDOW: ("hours",),
    DirectiveType.NO_DISCHARGE_WINDOW: ("hours",),
    DirectiveType.MAX_GRID_WINDOW: ("hours", "max_grid_kwh"),
}


def _require(cond: bool, reason: str, message: str) -> None:
    if not cond:
        raise GuardrailError(reason, message)


def _is_int(value: Any) -> bool:
    # bool is a subclass of int in Python; reject booleans masquerading as ints.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real_number(value: Any) -> bool:
    # Accept int or float (but not bool) and require finiteness.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_hours(hours: Any) -> None:
    if not isinstance(hours, list) or len(hours) == 0:
        raise GuardrailError(
            "hours_must_be_nonempty_list",
            f"'hours' must be a non-empty list, got {type(hours).__name__}",
        )
    for idx, h in enumerate(hours):
        if not _is_int(h):
            raise GuardrailError(
                "hours_must_be_ints",
                f"'hours[{idx}]' must be an int, got {type(h).__name__}: {h!r}",
            )
        if h < 0 or h > 23:
            raise GuardrailError(
                "hours_out_of_range",
                f"'hours[{idx}]' = {h} is outside the valid 0..23 range",
            )
        if idx > 0 and h <= hours[idx - 1]:
            if h == hours[idx - 1]:
                raise GuardrailError(
                    "hours_duplicate",
                    f"'hours[{idx}]' = {h} duplicates 'hours[{idx - 1}]'",
                )
            raise GuardrailError(
                "hours_not_ascending",
                f"'hours[{idx}]' = {h} is not strictly greater than "
                f"'hours[{idx - 1}]' = {hours[idx - 1]}",
            )


def _validate_factor(factor: Any) -> None:
    if not _is_real_number(factor):
        raise GuardrailError(
            "factor_must_be_real_number",
            f"'factor' must be a finite int or float, got {factor!r}",
        )
    value = float(factor)
    if value < 0.0 or value > 1.0:
        raise GuardrailError(
            "factor_out_of_range",
            f"'factor' = {value} is outside the valid [0, 1] range",
        )


def _validate_nonneg_real(name: str, value: Any) -> None:
    if not _is_real_number(value):
        raise GuardrailError(
            f"{name}_must_be_real_number",
            f"'{name}' must be a finite int or float, got {value!r}",
        )
    if float(value) < 0.0:
        raise GuardrailError(
            f"{name}_negative",
            f"'{name}' = {value} must be non-negative",
        )


def _validate_structured_adjustment(
    directive_type: DirectiveType, adjustment: Any
) -> None:
    if not isinstance(adjustment, dict):
        raise GuardrailError(
            "structured_adjustment_must_be_dict",
            f"'structured_adjustment' must be a dict for {directive_type.value}, "
            f"got {type(adjustment).__name__}",
        )

    required = _REQUIRED_ADJUSTMENT_KEYS[directive_type]
    for key in required:
        if key not in adjustment:
            raise GuardrailError(
                "missing_required_key",
                f"{directive_type.value} requires key '{key}' in "
                f"structured_adjustment",
            )

    if directive_type == DirectiveType.SOLAR_REDUCTION:
        _validate_hours(adjustment["hours"])
        _validate_factor(adjustment["factor"])
    elif directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
        _validate_hours(adjustment["hours"])
        _validate_nonneg_real("minimum_energy_kwh", adjustment["minimum_energy_kwh"])
    elif directive_type in (
        DirectiveType.NO_CHARGE_WINDOW,
        DirectiveType.NO_DISCHARGE_WINDOW,
    ):
        _validate_hours(adjustment["hours"])
    elif directive_type == DirectiveType.MAX_GRID_WINDOW:
        _validate_nonneg_real("max_grid_kwh", adjustment["max_grid_kwh"])
        _validate_hours(adjustment["hours"])


def validate_directive(raw: dict, note_count: int) -> dict:
    """Validate an LLM-emitted directive dict.

    Mirrors :class:`schemas.DirectiveInterpretation` but rejects malformed
    payloads deterministically with a ``GuardrailError`` carrying a stable
    ``reason`` string. Returns a plain dict that is a drop-in for the Pydantic
    model (so test fixtures and downstream code can treat them identically).
    """
    if not isinstance(raw, dict):
        raise GuardrailError(
            "raw_must_be_dict",
            f"raw directive must be a dict, got {type(raw).__name__}",
        )

    # 1. note_index
    note_index = raw.get("note_index")
    if not _is_int(note_index):
        raise GuardrailError(
            "note_index_must_be_int",
            f"'note_index' must be an int, got {type(note_index).__name__}: "
            f"{note_index!r}",
        )
    if note_index < 0 or note_index >= note_count:
        raise GuardrailError(
            "note_index_out_of_range",
            f"'note_index' = {note_index} is outside valid range 0..{note_count - 1}",
        )

    # 2. directive_type
    raw_type = raw.get("directive_type")
    if raw_type not in {dt.value for dt in DirectiveType}:
        raise GuardrailError(
            "unknown_directive_type",
            f"'directive_type' = {raw_type!r} is not one of "
            f"{sorted(dt.value for dt in DirectiveType)}",
        )
    directive_type = DirectiveType(raw_type)

    # 3. applies (raw, before type-specific rules)
    applies = raw.get("applies")
    if not isinstance(applies, bool):
        raise GuardrailError(
            "applies_must_be_bool",
            f"'applies' must be a bool, got {type(applies).__name__}: {applies!r}",
        )

    adjustment = raw.get("structured_adjustment")

    # 4. no_op requires applies=False and structured_adjustment=None
    if directive_type == DirectiveType.NO_OP:
        if applies:
            raise GuardrailError(
                "no_op_with_applies_true",
                "directive_type 'no_op' requires 'applies' to be false",
            )
        if adjustment is not None:
            raise GuardrailError(
                "no_op_with_adjustment",
                "directive_type 'no_op' requires 'structured_adjustment' to be null",
            )
    else:
        if not applies:
            raise GuardrailError(
                "non_no_op_with_applies_false",
                f"directive_type '{directive_type.value}' requires 'applies' to be true",
            )
        if adjustment is None:
            raise GuardrailError(
                "non_no_op_missing_adjustment",
                f"directive_type '{directive_type.value}' requires a non-null "
                f"'structured_adjustment'",
            )
        _validate_structured_adjustment(directive_type, adjustment)

    # 5. explanation (optional but, if present, must be a string)
    explanation = raw.get("explanation", "")
    if not isinstance(explanation, str):
        raise GuardrailError(
            "explanation_must_be_string",
            f"'explanation' must be a string when present, got "
            f"{type(explanation).__name__}",
        )

    return {
        "note_index": note_index,
        "applies": applies,
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }
