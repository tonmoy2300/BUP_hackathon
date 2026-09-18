from __future__ import annotations

import pytest

from guardrails import GuardrailError, validate_directive


# A minimal "valid" baseline that every non-no_op test mutates locally.
def _base_no_op() -> dict:
    return {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "ok",
    }


def _base_solar_reduction() -> dict:
    return {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [10, 11, 12], "factor": 0.5},
        "explanation": "halve solar",
    }


# ---------- happy paths ----------


def test_valid_no_op_passes() -> None:
    result = validate_directive(_base_no_op(), note_count=2)
    assert result["directive_type"] == "no_op"
    assert result["applies"] is False
    assert result["structured_adjustment"] is None


@pytest.mark.parametrize(
    "raw",
    [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [0], "factor": 0.0}, "explanation": "off"},
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [0], "factor": 1.0}, "explanation": "full"},
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [10, 11], "factor": 0.5,
                                    "note": "extra keys are fine"},
         "explanation": "extra"},
        {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [18, 19], "minimum_energy_kwh": 0},
         "explanation": "zero"},
        {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [18, 19], "minimum_energy_kwh": 12.5},
         "explanation": "half"},
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "structured_adjustment": {"hours": [0, 1, 2]}, "explanation": "morning"},
        {"note_index": 1, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"hours": [22, 23]}, "explanation": "evening"},
        {"note_index": 0, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 5.0},
         "explanation": "evening cap"},
    ],
)
def test_valid_non_no_op_variants_pass(raw: dict) -> None:
    result = validate_directive(raw, note_count=2)
    assert result["applies"] is True
    assert result["structured_adjustment"] is not None


# ---------- adversarial: directive_type ----------


def test_unknown_directive_type_rejected() -> None:
    raw = {**_base_no_op(), "directive_type": "explode_reactor"}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "unknown_directive_type"


def test_non_string_directive_type_rejected() -> None:
    raw = {**_base_no_op(), "directive_type": 42}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "unknown_directive_type"


# ---------- adversarial: no_op vs applies ----------


def test_no_op_with_applies_true_rejected() -> None:
    raw = {**_base_no_op(), "applies": True}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "no_op_with_applies_true"


def test_non_no_op_with_applies_false_rejected() -> None:
    raw = {**_base_solar_reduction(), "applies": False}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "non_no_op_with_applies_false"


def test_no_op_with_structured_adjustment_rejected() -> None:
    raw = {**_base_no_op(), "structured_adjustment": {"factor": 0.5}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "no_op_with_adjustment"


# ---------- adversarial: missing structured_adjustment ----------


@pytest.mark.parametrize(
    "directive_type",
    ["solar_reduction", "minimum_battery_reserve", "no_charge_window",
     "no_discharge_window", "max_grid_window"],
)
def test_non_no_op_missing_adjustment_rejected(directive_type: str) -> None:
    raw = {
        "note_index": 0,
        "applies": True,
        "directive_type": directive_type,
        "structured_adjustment": None,
        "explanation": "",
    }
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "non_no_op_missing_adjustment"


def test_solar_reduction_missing_factor_rejected() -> None:
    raw = {**_base_solar_reduction(), "structured_adjustment": {}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "missing_required_key"


def test_max_grid_window_missing_hours_rejected() -> None:
    raw = {
        "note_index": 0,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {"max_grid_kwh": 5.0},
        "explanation": "",
    }
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "missing_required_key"


def test_max_grid_window_missing_max_grid_rejected() -> None:
    raw = {
        "note_index": 0,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": [0, 1, 2]},
        "explanation": "",
    }
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "missing_required_key"


# ---------- adversarial: hours shape ----------


def test_hours_out_of_order_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [5, 3, 7]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_not_ascending"


def test_hours_duplicate_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [3, 3, 4]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_duplicate"


def test_hours_equal_boundary_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [5, 5]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    # Equal counts as duplicate, which is the more specific reason.
    assert exc.value.reason == "hours_duplicate"


def test_hours_value_24_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [23, 24]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_out_of_range"


def test_hours_value_negative_one_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [-1, 0]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_out_of_range"


def test_hours_must_be_ints() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": [1.5, 2]}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_must_be_ints"


def test_hours_must_be_nonempty() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "no_charge_window",
           "structured_adjustment": {"hours": []}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_must_be_nonempty_list"


def test_solar_reduction_hours_duplicate_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10, 10, 11], "factor": 0.5}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_duplicate"


def test_solar_reduction_hours_out_of_range_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10, 24], "factor": 0.5}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "hours_out_of_range"


# ---------- adversarial: factor ----------


def test_factor_above_one_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10], "factor": 1.5}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "factor_out_of_range"


def test_factor_negative_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10], "factor": -0.1}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "factor_out_of_range"


def test_factor_non_numeric_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10], "factor": "half"}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason in {"factor_must_be_real_number", "missing_required_key"}


def test_factor_infinity_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10], "factor": float("inf")}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "factor_must_be_real_number"


def test_factor_nan_rejected() -> None:
    raw = {**_base_solar_reduction(),
           "structured_adjustment": {"hours": [10], "factor": float("nan")}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "factor_must_be_real_number"


def test_solar_reduction_missing_hours_rejected() -> None:
    raw = {**_base_solar_reduction(), "structured_adjustment": {"factor": 0.5}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "missing_required_key"


# ---------- adversarial: energy caps ----------


def test_minimum_energy_negative_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "minimum_battery_reserve",
           "structured_adjustment": {"hours": [18], "minimum_energy_kwh": -1.0}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "minimum_energy_kwh_negative"


def test_minimum_energy_infinity_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "minimum_battery_reserve",
           "structured_adjustment": {"hours": [18], "minimum_energy_kwh": float("inf")}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "minimum_energy_kwh_must_be_real_number"


def test_minimum_battery_reserve_missing_hours_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "minimum_battery_reserve",
           "structured_adjustment": {"minimum_energy_kwh": 5.0}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "missing_required_key"


def test_max_grid_negative_rejected() -> None:
    raw = {**_base_no_op(),
           "applies": True,
           "directive_type": "max_grid_window",
           "structured_adjustment": {"hours": [18], "max_grid_kwh": -0.1}}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=1)
    assert exc.value.reason == "max_grid_kwh_negative"


# ---------- adversarial: note_index ----------


def test_note_index_out_of_range_high() -> None:
    raw = {**_base_no_op(), "note_index": 3}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=2)
    assert exc.value.reason == "note_index_out_of_range"


def test_note_index_negative_rejected() -> None:
    raw = {**_base_no_op(), "note_index": -1}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=2)
    assert exc.value.reason == "note_index_out_of_range"


def test_note_index_non_int_rejected() -> None:
    raw = {**_base_no_op(), "note_index": "0"}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=2)
    assert exc.value.reason == "note_index_must_be_int"


def test_note_index_bool_rejected() -> None:
    # bool is a subclass of int in Python; explicitly rejected.
    raw = {**_base_no_op(), "note_index": True}
    with pytest.raises(GuardrailError) as exc:
        validate_directive(raw, note_count=2)
    assert exc.value.reason == "note_index_must_be_int"


def test_duplicate_note_index_across_two_entries_both_pass() -> None:
    """note_count constrains the index range, not uniqueness.

    Guardrails validates a single directive at a time, so two entries sharing
    the same note_index individually pass. Cross-entry uniqueness is the job
    of the higher-level orchestrator (later scoring logic) — this test
    pins that behavior so it doesn't accidentally change.
    """
    note_count = 2
    a = validate_directive(_base_no_op(), note_count=note_count)
    b = validate_directive(_base_no_op(), note_count=note_count)
    assert a["note_index"] == b["note_index"] == 0


# ---------- fixture: distractor scenario ----------


def test_all_no_op_with_one_distractor_fixture() -> None:
    """Two operator notes, only one is a real directive — guardrails-only view.

    This won't fail guardrails (each entry individually is well-formed), but
    it's preserved here as a fixture for the scoring logic that comes later.
    """
    raw_directives = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [14, 15], "factor": 0.7},
         "explanation": "real directive"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None,
         "explanation": "off-topic note, ignore"},
    ]
    validated = [validate_directive(d, note_count=2) for d in raw_directives]
    assert len(validated) == 2
    assert validated[0]["applies"] is True
    assert validated[1]["directive_type"] == "no_op"


# ---------- non-dict guard ----------


def test_raw_must_be_dict() -> None:
    with pytest.raises(GuardrailError) as exc:
        validate_directive("not a dict", note_count=1)  # type: ignore[arg-type]
    assert exc.value.reason == "raw_must_be_dict"
