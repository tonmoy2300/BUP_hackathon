"""test_final_validator.py — tests for final_validator.validate_schedule.

Passing case: solve SAMPLE-02 input with a hand-built no_charge_window
directive for hours [2, 3, 4] and verify validate_schedule passes.

Failing cases: one mutated schedule per check (9 checks, each expected to
raise FinalValidationError).
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from final_validator import FinalValidationError, validate_schedule
from optimizer import solve_schedule


# ---------------------------------------------------------------------------
# Load SAMPLE-02 input data.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)

with open(os.path.join(_HERE, "sample02_real.json"), encoding="utf-8") as _f:
    _SAMPLE02 = json.load(_f)

_HOURS = _SAMPLE02["hours"]
_BATTERY = _SAMPLE02["battery"]

# Hand-built no_charge_window directive for hours [2, 3, 4].
_DIRECTIVES = [
    {
        "note_index": 0,
        "applies": True,
        "directive_type": "no_charge_window",
        "structured_adjustment": {"hours": [2, 3, 4]},
        "explanation": "battery charger isolated 2-5 AM",
    }
]


@pytest.fixture(scope="module")
def valid_schedule():
    """Solve SAMPLE-02 with the no_charge_window directive and return the schedule."""
    sched = solve_schedule(_HOURS, _BATTERY, _DIRECTIVES)
    return sched


# ---------------------------------------------------------------------------
# Passing case
# ---------------------------------------------------------------------------

def test_valid_schedule_passes(valid_schedule):
    """validate_schedule must NOT raise on the correctly solved schedule."""
    validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, valid_schedule)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mutated(valid_schedule, hour: int, **overrides) -> dict:
    """Return a deep-copied schedule with specific fields mutated for one hour."""
    sched = copy.deepcopy(valid_schedule)
    entry = next(e for e in sched["hourly_plan"] if e["hour"] == hour)
    for key, val in overrides.items():
        entry[key] = val
    return sched


# ---------------------------------------------------------------------------
# Failing case 1: energy balance violated
# ---------------------------------------------------------------------------

def test_energy_balance_violated(valid_schedule):
    """Pumping extra grid kWh into a single hour breaks the energy balance."""
    sched = _mutated(valid_schedule, hour=5, grid_kwh=9999.0)
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 2: solar_used exceeds effective solar
# ---------------------------------------------------------------------------

def test_solar_used_exceeds_effective_solar(valid_schedule):
    """solar_used above effective_solar (no directive reduction here) must fail."""
    # hour 12 has solar_kwh=110 in SAMPLE-02; claim 999 used.
    sched = _mutated(valid_schedule, hour=12, solar_used_kwh=999.0)
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 3: battery energy above capacity
# ---------------------------------------------------------------------------

def test_battery_energy_above_capacity(valid_schedule):
    """battery_energy_after above capacity must fail."""
    sched = _mutated(valid_schedule, hour=8, battery_energy_after_kwh=9999.0)
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 4: charge exceeds max_charge_per_hour
# ---------------------------------------------------------------------------

def test_charge_exceeds_max_per_hour(valid_schedule):
    """battery_kwh (charge) above max_charge_kwh_per_hour must fail."""
    sched = _mutated(
        valid_schedule,
        hour=1,
        battery_action="charge",
        battery_kwh=9999.0,
    )
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 5: charge in no_charge_window hours
# ---------------------------------------------------------------------------

def test_charge_in_no_charge_hour(valid_schedule):
    """Any nonzero charge in a no_charge_window hour must fail."""
    # hours 2, 3, 4 are no_charge per _DIRECTIVES.
    # Force a tiny charge of 1 kWh in hour 3.
    sched = _mutated(
        valid_schedule,
        hour=3,
        battery_action="charge",
        battery_kwh=1.0,
    )
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 6: grid exceeds max_grid_window cap
# (use a bespoke directive set that includes a max_grid_window)
# ---------------------------------------------------------------------------

def test_grid_exceeds_cap(valid_schedule):
    """grid_kwh above a max_grid_window cap must fail."""
    # Use a standalone max_grid_window directive with a generous cap so the LP
    # remains feasible.  hour 11 has demand=175, solar=100, so the uncapped
    # optimal grid draw is ~75 kWh; capping at 120 kWh is feasible.
    directives_cap_only = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [11], "max_grid_kwh": 120.0},
            "explanation": "grid capped at 120 kWh in hour 11",
        }
    ]
    sched_capped = solve_schedule(_HOURS, _BATTERY, directives_cap_only)
    # Verify the schedule itself passes before mutating.
    validate_schedule(_HOURS, _BATTERY, directives_cap_only, sched_capped)
    # Now mutate hour 11 grid_kwh to above the cap.
    sched_bad = _mutated(sched_capped, hour=11, grid_kwh=999.0)
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, directives_cap_only, sched_bad)


# ---------------------------------------------------------------------------
# Failing case 7: energy transition chain broken
# ---------------------------------------------------------------------------

def test_energy_transition_broken(valid_schedule):
    """Setting battery_energy_after to a wrong value breaks the chain check."""
    sched = _mutated(valid_schedule, hour=6, battery_energy_after_kwh=0.0)
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 8: total_grid_kwh doesn't match hourly sum
# ---------------------------------------------------------------------------

def test_total_grid_kwh_mismatch(valid_schedule):
    """Lying about total_grid_kwh must fail the totals check."""
    sched = copy.deepcopy(valid_schedule)
    sched["total_grid_kwh"] = 0.0
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)


# ---------------------------------------------------------------------------
# Failing case 9: total_cost_bdt doesn't match sum(grid * tariff)
# ---------------------------------------------------------------------------

def test_total_cost_bdt_mismatch(valid_schedule):
    """Lying about total_cost_bdt must fail the cost check."""
    sched = copy.deepcopy(valid_schedule)
    sched["total_cost_bdt"] = 0.0
    with pytest.raises(FinalValidationError):
        validate_schedule(_HOURS, _BATTERY, _DIRECTIVES, sched)
