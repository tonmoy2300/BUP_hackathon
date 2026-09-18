# Gridwise Energy Optimizer

A FastAPI micro-service that interprets free-text operator notes with a Groq
LLM, validates the resulting directives through a guardrail layer, solves a
linear-programming dispatch schedule with PuLP/CBC, and verifies the output
before returning a 24-hour battery + grid plan.

---

## Pipeline

```
Energy Data + Operator Notes
        │
        ▼
 ┌──────────────────┐
 │  LLM Interpreter │  (Groq — qwen/qwen3.8-27b, tool-call mode)
 └──────────────────┘
        │  raw directive list
        ▼
 ┌──────────────────┐
 │ Guardrail        │  (guardrails.py + orchestrator.py)
 │ Validator        │  structural & semantic checks; raises GuardrailError
 └──────────────────┘
        │  validated directive list
        ▼
 ┌──────────────────┐
 │  LP Optimizer    │  (optimizer.py — PuLP/CBC, 24-variable LP)
 └──────────────────┘
        │  hourly schedule
        ▼
 ┌──────────────────┐
 │ Final Validator  │  (final_validator.py) replays & checks all invariants
 └──────────────────┘
        │
        ▼
     Response  (JSON — 24-hour hourly_plan + totals)
```

---

## Model and reasoning_effort

**Model:** `qwen/qwen3.8-27b` (override with `GROQ_MODEL`).

The model is called in **tool-call mode** with `reasoning_effort="none"`.
This removes the model's extended chain-of-thought budget, cutting median
latency from ~10 s to ~2–3 s per request with no measurable accuracy loss for
the structured directive extraction task — the schema and rich prompt
examples already supply all the reasoning scaffolding the model needs.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **yes** | Groq API key. Keep this out of source control. |
| `GROQ_MODEL` | no | Override the Groq model string. Default: `qwen/qwen3.8-27b`. |

Copy `.env.example` to `.env` and fill in your key — the server loads it
automatically via `python-dotenv`.

---

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Browse the interactive Swagger UI at <http://127.0.0.1:8000/docs>.

---

## Docker

**Build:**

```bash
docker build -t {{DOCKER_IMAGE}} .
```

**Run:**

```bash
docker run --env-file .env -p 8000:8000 {{DOCKER_IMAGE}}
```

The server listens on port 8000 inside the container.

---

## curl examples

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Optimize energy (using sample02_real.json)

```bash
curl -X POST {{PUBLIC_URL}}/optimize-energy \
     -H "Content-Type: application/json" \
     -d @sample02_real.json
```

---

## Directive types

The LLM maps each operator note to exactly one of six directive types:

| Directive type | Effect |
|---|---|
| `solar_reduction` | Reduces usable solar output by a factor during specified hours. `factor` is the **remaining** fraction (e.g. 80 % cut → factor 0.2). |
| `minimum_battery_reserve` | Forces the battery to stay above a minimum energy level (kWh) during specified hours. |
| `no_charge_window` | Forbids battery charging during specified hours. |
| `no_discharge_window` | Forbids battery discharging during specified hours. |
| `max_grid_window` | Caps grid import at a maximum kWh figure during specified hours. |
| `no_op` | Note is off-topic / non-actionable; no schedule constraint is applied. |

Hour windows are **start-inclusive, end-exclusive** (e.g. "1 PM until 3 PM" → hours [13, 14]).

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | HTTP framework and request/response validation |
| `uvicorn` | ASGI server |
| `pydantic >= 2` | Strict schema models |
| `pulp` | LP formulation and CBC solver interface |
| `groq` | Groq Python SDK |
| `python-dotenv` | `.env` file loading |
| `httpx` | Required by FastAPI TestClient |

---

## Known limitations

- **LLM ambiguity on AM/PM** — When a note omits AM/PM and the hour is
  ambiguous (e.g. "from ten until twelve"), the model infers daytime context
  but can occasionally resolve to the wrong 12-hour period.  The solar
  cross-check hint mitigates this but doesn't fully eliminate it.
- **Groq rate limits** — High request volume may hit Groq's rate limits and
  cause transient 500 errors.  The interpreter retries once automatically
  (total two attempts), but sustained traffic requires a queue or back-off
  strategy.
- **Single-day horizon** — The LP optimises a single 24-hour window.
  Multi-day battery carry-over, forecasting uncertainty, and demand-response
  programmes are out of scope.
