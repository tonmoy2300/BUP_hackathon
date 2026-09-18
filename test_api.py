"""test_api.py — FastAPI integration tests using TestClient.

interpret_notes is mocked throughout so no real Groq call is made.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# A pre-built directive list that passes guardrails (one no_op directive).
# ---------------------------------------------------------------------------

_NO_OP_DIRECTIVE = [
    {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "off-topic note",
    }
]

# A no_charge_window directive (for notes that describe real directives).
_NO_CHARGE_DIRECTIVE = [
    {
        "note_index": 0,
        "applies": True,
        "directive_type": "no_charge_window",
        "structured_adjustment": {"hours": [2, 3, 4]},
        "explanation": "charger isolated 2-5 AM",
    }
]

# ---------------------------------------------------------------------------
# Load a valid payload from sample02_real.json.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)

with open(os.path.join(_HERE, "sample02_real.json"), encoding="utf-8") as _f:
    _VALID_PAYLOAD = json.load(_f)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _post(payload, mock_return=None):
    if mock_return is None:
        mock_return = _NO_OP_DIRECTIVE
    with patch("main.interpret_notes", return_value=mock_return):
        return client.post("/optimize-energy", json=payload)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 200 on a valid request
# ---------------------------------------------------------------------------

def test_valid_request_returns_200():
    resp = _post(_VALID_PAYLOAD, mock_return=_NO_CHARGE_DIRECTIVE)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == "SAMPLE-02"
    assert len(data["hourly_plan"]) == 24
    assert "total_grid_kwh" in data


# ---------------------------------------------------------------------------
# 400 on a missing required field
# ---------------------------------------------------------------------------

def test_missing_field_returns_400():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "battery"}
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Malformed request."}


# ---------------------------------------------------------------------------
# 400 on a wrong type
# ---------------------------------------------------------------------------

def test_wrong_type_returns_400():
    payload = {**_VALID_PAYLOAD, "operator_notes": "not a list"}
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Malformed request."}


# ---------------------------------------------------------------------------
# 400 on duplicate hour values
# ---------------------------------------------------------------------------

def test_duplicate_hour_values_returns_400():
    # Duplicate hour 0 (replace hour 1 entry with another hour 0).
    hours = [
        h if h["hour"] != 1 else {**h, "hour": 0}
        for h in _VALID_PAYLOAD["hours"]
    ]
    payload = {**_VALID_PAYLOAD, "hours": hours}
    resp = _post(payload)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Malformed request."}


# ---------------------------------------------------------------------------
# 400 on a blank (whitespace-only) operator note
# ---------------------------------------------------------------------------

def test_blank_note_returns_400():
    # A note that is only whitespace must be rejected before the LLM is called.
    payload = {**_VALID_PAYLOAD, "operator_notes": ["   "]}
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Malformed request."}

