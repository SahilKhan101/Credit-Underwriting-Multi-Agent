"""
Tests for the Report Generation Agent, specifically the narrative section.
Run: .venv/bin/pytest tests/test_report_agent.py -v
"""
import pytest
from agents.report_agent import report_generation_node, _score_breakdown_table


# ── Minimal fake state that covers every field the agent reads ─────────────────

MOCK_STATE = {
    "application_id": "TEST-001",
    "applicant_profile": {
        "name": "Test Applicant",
        "age": 35,
        "employment_type": "SALARIED",
        "monthly_income": 80_000,
        "city": "Mumbai",
        "loan_amount": 1_200_000,
        "loan_tenure_months": 36,
        "loan_purpose": "PERSONAL",
    },
    "bureau_analysis": {
        "bureau_score": 680.0,
        "summary": "CIBIL 720, low utilisation, no DPD",
        "flags": ["HIGH FOIR: 42%"],
    },
    "income_analysis": {
        "income_score": 650.0,
        "summary": "Salaried, stable salary credits for 6 months",
        "flags": [],
    },
    "bank_statement_analysis": {
        "cashflow_score": 700.0,
        "summary": "Low bounce rate, ABB/EMI ratio 2.1x",
        "flags": [],
    },
    "fraud_signals": {
        "fraud_score": 50.0,
        "summary": "No velocity flags or cash anomalies",
        "flags": [],
    },
    "policy_check": {
        "policy_score": 900.0,
        "eligible": True,
        "summary": "Meets all RBI and internal policy criteria",
        "flags": [],
    },
    "credit_score": 728.0,
    "recommendation": "APPROVE_WITH_CONDITIONS",
    "confidence": 0.88,
    "decision_rationale": "Strong cashflow with mild FOIR concern.",
    "key_strengths": ["Low fraud risk", "Stable salary"],
    "key_risks": ["FOIR at 42% — near limit"],
    "conditions": ["Provide last 3 months payslips"],
    "agent_trace": ["[Bureau Agent] done", "[Decision Agent] done"],
}


def test_score_breakdown_table():
    table = _score_breakdown_table(680, 650, 700, 50, 900, 728)
    assert "Bureau Analysis" in table
    assert "728" in table
    assert "|" in table   # markdown table format


def test_report_node_returns_memo():
    result = report_generation_node(MOCK_STATE)
    assert "credit_memo" in result
    assert len(result["credit_memo"]) > 100


def test_memo_contains_required_sections():
    result = report_generation_node(MOCK_STATE)
    memo = result["credit_memo"]
    assert "Credit Assessment Memo" in memo
    assert "Score Breakdown" in memo
    assert "Underwriting Narrative" in memo
    assert "Key Decision Factors" in memo
    assert "Agent Audit Trail" in memo


def test_memo_contains_applicant_details():
    result = report_generation_node(MOCK_STATE)
    memo = result["credit_memo"]
    assert "TEST-001" in memo
    assert "Test Applicant" in memo
    assert "728" in memo   # credit score


def test_narrative_is_llm_generated():
    """Narrative should NOT contain the fallback error string when LLM is working."""
    result = report_generation_node(MOCK_STATE)
    memo = result["credit_memo"]
    narrative_section = memo.split("## Underwriting Narrative")[1].split("---")[0].strip()
    assert "narrative unavailable" not in narrative_section.lower(), (
        f"LLM narrative failed. Got fallback text:\n{narrative_section}"
    )
    assert len(narrative_section) > 100, "Narrative is too short — LLM may have returned empty"


def test_agent_trace_appended():
    result = report_generation_node(MOCK_STATE)
    trace = result.get("agent_trace", [])
    assert any("Report Agent" in t for t in trace)
