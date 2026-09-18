from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from guardrails import GuardrailError
from llm_interpreter import LLMInterpretationError, interpret_notes
from orchestrator import validate_directive_list
from schemas import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Gridwise Energy Optimizer", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _placeholder_hourly_plan(request: OptimizeRequest) -> list[HourlyPlanEntry]:
    """Idle the battery with zero grid import for all 24 hours.

    Not touching optimization yet -- this is unchanged placeholder logic.
    """
    battery_start = request.battery.initial_energy_kwh
    return [
        HourlyPlanEntry(
            hour=h,
            grid_kwh=0.0,
            solar_used_kwh=0.0,
            battery_action="idle",
            battery_kwh=0.0,
            battery_energy_after_kwh=battery_start,
        )
        for h in range(24)
    ]


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(request: OptimizeRequest) -> OptimizeResponse:
    try:
        raw_directives = interpret_notes(request.operator_notes)
        validated = validate_directive_list(
            raw_directives, note_count=len(request.operator_notes)
        )
    except LLMInterpretationError as exc:
        logger.error(
            "LLM interpretation failed for scenario_id=%r: %s",
            request.scenario_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to process operator notes. Please try again.",
        ) from None
    except GuardrailError as exc:
        logger.error(
            "directive validation failed for scenario_id=%r: reason=%s",
            request.scenario_id,
            exc.reason,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to process operator notes. Please try again.",
        ) from None

    directive_interpretation = [DirectiveInterpretation(**entry) for entry in validated]
    hourly_plan = _placeholder_hourly_plan(request)

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directive_interpretation,
        hourly_plan=hourly_plan,
        total_grid_kwh=0.0,
        total_cost_bdt=0.0,
        peak_grid_kwh=0.0,
        plan_summary=(
            "Placeholder plan: no optimization applied. Battery remains at its "
            "initial state across all 24 hours; no grid import, no solar dispatch. "
            "Use this response to validate the request/response contract shape."
        ),
    )
