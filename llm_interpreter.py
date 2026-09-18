from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


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

Interpret each operator note independently of the others -- one note's \
content must never change how you interpret a different note. A note that \
is clearly irrelevant to today's energy schedule is no_op; do not force an \
irrelevant note into one of the five real directive types.

Call the emit_directive_interpretations tool exactly once. Its `directives` \
array must contain exactly one entry per operator note, in note_index order.
"""


def _build_messages(operator_notes: list[str]) -> list[dict]:
    notes_block = "\n".join(
        f"{i}: {note}" for i, note in enumerate(operator_notes)
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Operator notes (note_index: text), one entry required per "
                f"note:\n{notes_block}"
            ),
        },
    ]


def interpret_notes(operator_notes: list[str]) -> list[dict]:
    """Ask the LLM to interpret a batch of operator notes into raw directives.

    Returns the raw list of directive dicts exactly as emitted by the model,
    in whatever order/shape it produced -- no validation is performed here.
    Validating and enforcing note_index coverage is orchestrator.py's job
    downstream (see validate_directive_list).

    Raises LLMInterpretationError, with a safe generic message, if the API
    call fails or the response can't be parsed into a directives list.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            reasoning_effort="none",
            tool_choice={
                "type": "function",
                "function": {"name": "emit_directive_interpretations"},
            },
            tools=[EMIT_DIRECTIVE_INTERPRETATIONS_TOOL],
            messages=_build_messages(operator_notes),
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
