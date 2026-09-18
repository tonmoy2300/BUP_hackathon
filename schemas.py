from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class HourEntry(BaseModel):
    model_config = ConfigDict(strict=True)

    hour: int
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float


class Battery(BaseModel):
    model_config = ConfigDict(strict=True)

    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    scenario_id: str
    operator_notes: list[str] = Field(..., min_length=1, max_length=3)
    hours: list[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes", mode="after")
    @classmethod
    def notes_must_not_be_blank(cls, notes: list[str]) -> list[str]:
        """Reject any note that is empty or whitespace-only after stripping."""
        for i, note in enumerate(notes):
            if not note.strip():
                raise ValueError(
                    f"operator_notes[{i}] must not be blank or whitespace-only"
                )
        return notes


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(strict=True)

    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    model_config = ConfigDict(strict=True)

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation] = Field(
        ..., min_length=1, max_length=3
    )
    hourly_plan: list[HourlyPlanEntry] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
