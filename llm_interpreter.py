from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Model selection
# Both candidate model strings were probed at development time:
#   "qwen/qwen3.6-27b" → NotFoundError (404) on this Groq account.
#   "qwen/qwen3.8-27b" → succeeds.
# The live probe confirmed "qwen/qwen3.8-27b" as the working default.
# Override at runtime with the GROQ_MODEL environment variable.
# ---------------------------------------------------------------------------
MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


class LLMInterpretationError(RuntimeError):
    """Raised when the LLM call or its response can't be turned into directives.

    Message is intentionally generic -- never includes raw API exception
    text or any key material. Callers/logs upstream can add their own
    context if needed; this error itself must stay safe to surface.
    """


_TOOL_DESCRIPTION = """\
Emit the directive interpretation for every operator note in this batch.

Call this exactly once, passing one entry in `directives` per operator note,
in note_index order (0-based, matching the note's position in the input list).

Each entry's `structured_adjustment` shape depends on its `directive_type`
(JSON Schema can't express that dependency directly, so it's spelled out
here -- follow it exactly):

- solar_reduction: {"hours": [int, ...], "factor": float}
  `hours` are the affected hours (0-23, unique, ascending). `factor` is the
  USABLE FRACTION OF SOLAR REMAINING after the reduction (e.g. an 80%
  reduction means factor = 0.2), in [0, 1].
- minimum_battery_reserve: {"hours": [int, ...], "minimum_energy_kwh": float}
  `hours` are the hours the reserve applies to. `minimum_energy_kwh` is the
  absolute minimum battery energy (kWh) that must remain available in those
  hours, non-negative.
- no_charge_window: {"hours": [int, ...]}
  `hours` are the hours battery charging is forbidden in.
- no_discharge_window: {"hours": [int, ...]}
  `hours` are the hours battery discharging is forbidden in.
- max_grid_window: {"hours": [int, ...], "max_grid_kwh": float}
  `hours` are the hours the cap applies to. `max_grid_kwh` is the maximum
  grid import (kWh) allowed in each of those hours, non-negative.
- no_op: structured_adjustment MUST be null, and applies MUST be false.

For every directive_type other than no_op, applies MUST be true.
"""

EMIT_DIRECTIVE_INTERPRETATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_directive_interpretations",
        "description": _TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "directives": {
                    "type": "array",
                    "description": (
                        "One interpretation entry per operator note, in "
                        "note_index order."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "note_index": {
                                "type": "integer",
                                "description": (
                                    "0-based index of the operator note this "
                                    "entry interprets."
                                ),
                            },
                            "directive_type": {
                                "type": "string",
                                "enum": [
                                    "solar_reduction",
                                    "minimum_battery_reserve",
                                    "no_charge_window",
                                    "no_discharge_window",
                                    "max_grid_window",
                                    "no_op",
                                ],
                            },
                            "applies": {
                                "type": "boolean",
                                "description": (
                                    "false for no_op, true for every other "
                                    "directive_type."
                                ),
                            },
                            "structured_adjustment": {
                                "type": ["object", "null"],
                                "description": (
                                    "Shape depends on directive_type -- see "
                                    "the tool description. Null iff "
                                    "directive_type is no_op."
                                ),
                            },
                            "explanation": {
                                "type": "string",
                                "description": (
                                    "Brief human-readable explanation of "
                                    "this interpretation."
                                ),
                            },
                        },
                        "required": [
                            "note_index",
                            "directive_type",
                            "applies",
                            "structured_adjustment",
                            "explanation",
                        ],
                    },
                },
            },
            "required": ["directives"],
        },
    },
}

_SYSTEM_PROMPT = """\
You interpret operator notes for a campus energy-optimization system into \
structured directives.

There are exactly six directive types:

- solar_reduction: usable solar output is reduced during some hours (e.g. \
panel cleaning, shading, partial outage). structured_adjustment.factor is \
the fraction of forecast solar that remains USABLE after the reduction -- \
not the fraction removed. An 80% reduction means factor = 0.2. A note \
saying solar is cut in half means factor = 0.5.
- minimum_battery_reserve: the battery must be kept above some minimum \
energy level (kWh) during some hours, e.g. for emergency/backup power. \
Convert relative language (e.g. "50% of capacity") into an absolute kWh \
figure using the battery's stated capacity if given in the note; if the \
note gives an absolute kWh figure directly, use that.
- no_charge_window: the battery must not charge during some hours (e.g. \
charger maintenance, isolation, hardware fault).
- no_discharge_window: the battery must not discharge during some hours \
(e.g. protection testing, isolation).
- max_grid_window: grid import is capped at some kWh figure during some \
hours (e.g. a feeder or transformer limit).
- no_op: the note does not describe an actionable energy-schedule directive \
for today's 24-hour horizon (e.g. it's about something unrelated, a future \
date, administrative trivia, or simply doesn't map to any of the five real \
directive types above). applies is false and structured_adjustment is null.

Hour windows are always start-inclusive, end-exclusive: "1 PM until 3 PM" \
means hours [13, 14] (hour 15 is NOT included). "from 2 AM until 5 AM" means \
hours [2, 3, 4]. List hours as unique integers 0-23 in ascending order.

When a note gives a clock time without AM/PM, infer the correct 24-hour \
value from context -- operational activities like panel washing, \
maintenance, and deliveries happen during standard daytime/business hours \
(roughly 6-22), not the middle of the night. Cross-check: if you're \
interpreting a solar_reduction directive, the affected hours should have \
nonzero solar_kwh in the provided data -- if they don't, reconsider your \
hour interpretation.

Examples:

Note: "Solar output will drop to about 20% from 1 PM to 3 PM"
-> solar_reduction, hours [13, 14], factor 0.2

Note: "Do not charge the battery from 2 PM to 4 PM."
-> no_charge_window, hours [14, 15]

Note: "Keep at least 120 kWh in reserve from 6 PM to 9 PM."
-> minimum_battery_reserve, hours [18, 19, 20], minimum_energy_kwh 120

Note: "The cafeteria updated next week's lunch menu."
-> no_op

The following three notes are different paraphrasings of the exact same \
solar directive -- all three must map to the identical interpretation, \
solar_reduction, hours [13, 14], factor 0.2:

Note: "Solar output will drop to about 20% from 1 PM to 3 PM"
Note: "Panel washing from one until three will leave roughly one-fifth of \
normal solar output"
Note: "Expect an 80% reduction in rooftop solar during the 1-3 PM \
maintenance window"

Notice the third example: "80% reduction" describes the fraction REMOVED, \
not remaining, so factor = 1 - 0.8 = 0.2 -- the same result as "drop to \
about 20%" and "one-fifth of normal." Different notes describing the same \
underlying directive must always resolve to the same structured_adjustment, \
regardless of wording or which numbers (percentage removed vs. percentage \
remaining vs. a fraction) the note happens to use.

Interpret each operator note independently of the others -- one note's \
content must never change how you interpret a different note. A note that \
is clearly irrelevant to today's energy schedule is no_op; do not force an \
irrelevant note into one of the five real directive types.

Call the emit_directive_interpretations tool exactly once. Its `directives` \
array must contain exactly one entry per operator note, in note_index order.
"""


def _build_messages(operator_notes: list[str], hours: list[dict] | None = None) -> list[dict]:
    notes_block = "\n".join(
        f"{i}: {note}" for i, note in enumerate(operator_notes)
    )
    user_content = (
        "Operator notes (note_index: text), one entry required per "
        f"note:\n{notes_block}"
    )

    if hours is not None:
        # Identify hours with nonzero solar to help the model cross-check.
        nonzero = sorted(
            h["hour"] for h in hours if float(h.get("solar_kwh", 0)) > 0
        )
        if nonzero:
            # Build compact range summary (e.g. "6-17") and per-hour values.
            run_start = nonzero[0]
            run_end = nonzero[-1]
            range_str = f"{run_start}-{run_end}" if run_start != run_end else str(run_start)
            per_hour_vals = ", ".join(
                f"h{h['hour']}={h['solar_kwh']}"
                for h in hours
                if float(h.get("solar_kwh", 0)) > 0
            )
            user_content += (
                f"\nHours with nonzero solar forecast: {range_str} "
                f"({per_hour_vals})"
            )
        else:
            user_content += "\nHours with nonzero solar forecast: none"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def interpret_notes(operator_notes: list[str], hours: list[dict] | None = None) -> list[dict]:
    """Ask the LLM to interpret a batch of operator notes into raw directives.

    Parameters
    ----------
    operator_notes:
        The free-text notes to interpret (1-3 strings).
    hours:
        Optional list of hourly dicts (each with at least ``hour`` and
        ``solar_kwh`` keys).  When provided, the user message is augmented
        with a list of which hours have nonzero solar forecast so the model
        can cross-check solar_reduction hour assignments.

    Returns the raw list of directive dicts exactly as emitted by the model,
    in whatever order/shape it produced -- no validation is performed here.
    Validating and enforcing note_index coverage is orchestrator.py's job
    downstream (see validate_directive_list).

    Raises LLMInterpretationError, with a safe generic message, if the API
    call fails or the response can't be parsed into a directives list.
    """
    try:
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
            timeout=8.0,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=MODEL,
            reasoning_effort="none",
            tool_choice={
                "type": "function",
                "function": {"name": "emit_directive_interpretations"},
            },
            tools=[EMIT_DIRECTIVE_INTERPRETATIONS_TOOL],
            messages=_build_messages(operator_notes, hours),
        )
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        directives = arguments["directives"]
    except Exception as exc:
        raise LLMInterpretationError(
            "Failed to interpret operator notes via the LLM."
        ) from exc

    if not isinstance(directives, list):
        raise LLMInterpretationError(
            "Failed to interpret operator notes via the LLM."
        )

    return directives
