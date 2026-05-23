"""
FastAPI REST API for the credit underwriting system.

Endpoints:
  POST /underwrite          — submit a loan application → returns full decision
  POST /underwrite/batch    — submit multiple applications (list)
  GET  /health              — health check
  GET  /applications/{id}   — retrieve cached result (in-memory for demo)

Run:
    uvicorn api.main:api --reload --port 8000
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import initial_state
from graph.workflow import app as workflow

api = FastAPI(
    title="CreditIQ — AI Underwriting API",
    description="Multi-agent credit underwriting powered by LangGraph + Claude",
    version="1.0.0",
)

# In-memory result cache (replace with DB/Redis in production)
_results: dict[str, dict] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ApplicantProfile(BaseModel):
    name: str
    age: int = Field(..., ge=18, le=70)
    employment_type: str = "SALARIED"
    monthly_income: float = Field(..., gt=0)
    employer: str = ""
    city: str = ""
    state: str = ""
    pin_code: str = ""
    loan_amount: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., ge=6, le=120)
    loan_purpose: str = "PERSONAL"
    requested_emi: float = Field(..., gt=0)


class BureauReport(BaseModel):
    cibil_score: int = Field(..., ge=300, le=900)
    total_accounts: int = 2
    active_accounts: int = 1
    credit_utilization: float = Field(..., ge=0, le=1)
    dpd_30_count: int = 0
    dpd_90_count: int = 0
    enquiries_last_6m: int = 0
    oldest_account_months: int = 12
    total_existing_emi: float = 0
    written_off_flag: bool = False
    secured_unsecured_ratio: float = 0.5


class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    type: str      # "credit" | "debit"
    balance_after: float = 0
    bounced: bool = False


class UnderwriteRequest(BaseModel):
    application_id: str | None = None
    applicant_profile: ApplicantProfile
    bureau_report: BureauReport
    bank_statements: list[Transaction] = Field(default_factory=list)


class UnderwriteResponse(BaseModel):
    application_id: str
    credit_score: float
    recommendation: str
    confidence: float
    decision_rationale: str
    key_strengths: list[str]
    key_risks: list[str]
    conditions: list[str]
    credit_memo: str
    agent_trace: list[str]
    errors: list[str]
    processing_time_ms: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@api.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "CreditIQ Underwriting API"}


@api.post("/underwrite", response_model=UnderwriteResponse)
async def underwrite(req: UnderwriteRequest) -> UnderwriteResponse:
    app_id = req.application_id or f"API-{uuid.uuid4().hex[:8].upper()}"

    state = initial_state(
        application_id    = app_id,
        applicant_profile = req.applicant_profile.model_dump(),
        bureau_report     = req.bureau_report.model_dump(),
        bank_statements   = [t.model_dump() for t in req.bank_statements],
    )

    try:
        result = workflow.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Underwriting failed: {exc}")

    response = UnderwriteResponse(
        application_id    = app_id,
        credit_score      = result.get("credit_score", 0),
        recommendation    = result.get("recommendation", "ERROR"),
        confidence        = result.get("confidence", 0),
        decision_rationale = result.get("decision_rationale", ""),
        key_strengths     = result.get("key_strengths", []),
        key_risks         = result.get("key_risks", []),
        conditions        = result.get("conditions", []),
        credit_memo       = result.get("credit_memo", ""),
        agent_trace       = result.get("agent_trace", []),
        errors            = result.get("errors", []),
        processing_time_ms = result.get("processing_time_ms", 0),
    )
    _results[app_id] = response.model_dump()
    return response


@api.post("/underwrite/batch")
async def underwrite_batch(requests: list[UnderwriteRequest]) -> list[dict[str, Any]]:
    results = []
    for req in requests:
        try:
            r = await underwrite(req)
            results.append(r.model_dump())
        except HTTPException as e:
            results.append({"error": e.detail})
    return results


@api.get("/applications/{application_id}")
async def get_application(application_id: str) -> dict:
    if application_id not in _results:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")
    return _results[application_id]
