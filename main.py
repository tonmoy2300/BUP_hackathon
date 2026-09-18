from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from guardrails import GuardrailError
from llm_interpreter import LLMInterpretationError, interpret_notes
from optimizer import OptimizationError, solve_schedule
from orchestrator import validate_directive_list
from schemas import (
    DirectiveInterpretation,
    HourlyPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Gridwise Energy Optimizer", version="0.1.0")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a flat 400 without leaking internal validation details."""
    logger.warning("Request validation error: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"detail": "Malformed request."},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all: log the real error server-side, return a safe 500."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(request: OptimizeRequest) -> OptimizeResponse:
    request_start = time.monotonic()

    # Verify the hour values are exactly {0..23} — no gaps, no duplicates.
    hour_values = [h.hour for h in request.hours]
    if sorted(hour_values) != list(range(24)):
        raise HTTPException(status_code=400, detail="Malformed request.")

    hours_data = [h.model_dump() for h in request.hours]

    # Try interpret + validate; retry once on LLMInterpretationError or
    # GuardrailError, but only if we are still within 12 s of the request
    # start (avoids blowing the 30 s budget when the first attempt itself
    # was slow).  Worst-case timing: 8 s (timeout) × 1 SDK retry = up to
    # 16 s for attempt 1; the main-level retry is skipped once elapsed ≥ 12 s.
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            raw_directives = interpret_notes(
                request.operator_notes, hours=hours_data
            )
            validated = validate_directive_list(
                raw_directives, note_count=len(request.operator_notes)
            )
            last_exc = None
            break
        except (LLMInterpretationError, GuardrailError) as exc:
            elapsed = time.monotonic() - request_start
            logger.warning(
                "attempt %d/%d failed for scenario_id=%r (elapsed=%.1fs): %s",
                attempt + 1,
                2,
                request.scenario_id,
                elapsed,
                exc,
            )
            last_exc = exc
            # Only allow the second attempt if time budget permits.
            if attempt == 0 and elapsed >= 12.0:
                logger.warning(
                    "skipping retry for scenario_id=%r: elapsed %.1fs >= 12 s",
                    request.scenario_id,
                    elapsed,
                )
                break

    if last_exc is not None:
        if isinstance(last_exc, LLMInterpretationError):
            logger.error(
                "LLM interpretation failed for scenario_id=%r: %s",
                request.scenario_id,
                last_exc,
            )
        else:
            logger.error(
                "directive validation failed for scenario_id=%r: reason=%s",
                request.scenario_id,
                last_exc.reason,  # type: ignore[union-attr]
            )
        raise HTTPException(
            status_code=500,
            detail="Unable to process operator notes. Please try again.",
        ) from None

    try:
        schedule = solve_schedule(
            hours_data,
            request.battery.model_dump(),
            validated,
        )
    except OptimizationError as exc:
        logger.error(
            "schedule optimization failed for scenario_id=%r: %s",
            request.scenario_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to process operator notes. Please try again.",
        ) from None

    # Final validation: replay the solver output and verify invariants.
    from final_validator import FinalValidationError, validate_schedule

    try:
        validate_schedule(hours_data, request.battery.model_dump(), validated, schedule)
    except FinalValidationError as exc:
        logger.error(
            "final validation failed for scenario_id=%r: %s",
            request.scenario_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error.",
        ) from None

    directive_interpretation = [DirectiveInterpretation(**entry) for entry in validated]
    hourly_plan = [HourlyPlanEntry(**entry) for entry in schedule["hourly_plan"]]

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directive_interpretation,
        hourly_plan=hourly_plan,
        total_grid_kwh=schedule["total_grid_kwh"],
        total_cost_bdt=schedule["total_cost_bdt"],
        peak_grid_kwh=schedule["peak_grid_kwh"],
        plan_summary=(
            "Optimized 24-hour battery/grid dispatch minimizing total grid cost, "
            "honoring all applicable operator directives and end-of-day battery "
            "neutrality."
        ),
    )
