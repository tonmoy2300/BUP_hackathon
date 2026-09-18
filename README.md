# Gridwise Energy Optimizer

FastAPI service that plans hourly battery + grid dispatch for a 24-hour horizon,
honoring operator directives parsed from free-text notes.

## Status

Scaffolding only. `/optimize-energy` currently returns a hardcoded placeholder
response so the request/response contract can be validated end-to-end. Real
optimization logic (PuLP MILP) is not yet wired in.

## Endpoints

- `GET /health` — liveness probe, returns `{"status": "ok"}`
- `POST /optimize-energy` — run a 24-hour dispatch plan (placeholder for now)

## Layout

```
gridwise/
├── main.py          # FastAPI app and route handlers
├── schemas.py       # Pydantic v2 request/response models (strict)
├── requirements.txt # fastapi, uvicorn, pydantic, pulp
└── README.md
```

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.
# BUP_hackathon
