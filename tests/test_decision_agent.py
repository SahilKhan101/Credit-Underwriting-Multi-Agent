"""
Tests for the Decision Agent — composite scoring and LLM structured output.
Run: .venv/bin/pytest tests/test_decision_agent.py -v
"""
import pytest
from agents.decision_agent import (
    decision_agent_node,
    _composite_score,
    _score_to_recommendation,
)


# ── Unit tests for pure scoring functions (no LLM) ────────────────────────────

def test_composite_score_all_perfect():
    score = _composite_score(1000, 1000, 1000, 0, 1000)
    assert score == 1000.0


def test_composite_score_all_zero():
    score = _composite_score(0, 0, 0, 1000, 0)
    assert score == 0.0


def test_composite_score_weights_sum():
    # With all components at 500, score should be ~500
    score = _composite_score(500, 500, 500, 500, 500)
    assert 490 < score < 510


def test_fraud_score_inverted():
    # High fraud (1000) should lower overall score vs low fraud (0)
    high_fraud = _composite_score(700, 700, 700, 1000, 700)
    low_fraud  = _composite_score(700, 700, 700, 0,    700)
    assert low_fraud > high_fraud


@pytest.mark.parametrize("score,eligible,expected", [
    (800, True,  "AUTO_APPROVE"),
    (700, True,  "APPROVE_WITH_CONDITIONS"),
    (600, True,  "REFER_TO_CREDIT_COMMITTEE"),
    (400, True,  "AUTO_REJECT"),
    (900, False, "AUTO_REJECT"),   # policy ineligible overrides score
])
def test_score_to_recommendation(score, eligible, expected):
    assert _score_to_recommendation(score, eligible) == expected


# ── Integration tests (call real LLM) ────────────────────────────────────────

MOCK_STATE = {
    "application_id": "TEST-002",
    "applicant_profile": {
        "name": "Test Applicant",
        "age": 35,
        "employment_type": "SALARIED",
        "monthly_income": 80_000,
        "loan_amount": 1_200_000,
        "loan_tenure_months": 36,
        "loan_purpose": "PERSONAL",
    },
    "bureau_analysis": {
        "bureau_score": 700.0,
        "summary": "CIBIL 730, moderate utilisation",
        "flags": [],
    },
    "income_analysis": {
        "income_score": 660.0,
        "summary": "Stable salaried income",
        "flags": [],
    },
    "bank_statement_analysis": {
        "cashflow_score": 720.0,
        "summary": "Healthy ABB, low bounce rate",
        "flags": [],
    },
    "fraud_signals": {
        "fraud_score": 30.0,
        "summary": "No fraud indicators",
        "flags": [],
    },
    "policy_check": {
        "policy_score": 900.0,
        "eligible": True,
        "summary": "Policy compliant",
        "flags": [],
    },
    "agent_trace": [],
    "errors": [],
}


def test_decision_node_returns_required_fields():
    result = decision_agent_node(MOCK_STATE)
    for field in ("credit_score", "recommendation", "confidence",
                  "decision_rationale", "key_strengths", "key_risks", "conditions"):
        assert field in result, f"Missing field: {field}"


def test_decision_node_score_in_range():
    result = decision_agent_node(MOCK_STATE)
    assert 0 <= result["credit_score"] <= 1000


def test_decision_node_recommendation_valid():
    result = decision_agent_node(MOCK_STATE)
    valid = {"AUTO_APPROVE", "APPROVE_WITH_CONDITIONS",
             "REFER_TO_CREDIT_COMMITTEE", "AUTO_REJECT"}
    assert result["recommendation"] in valid


def test_decision_node_llm_not_fallback():
    """LLM structured output should produce a non-empty rationale."""
    result = decision_agent_node(MOCK_STATE)
    assert "Rule-based decision" not in result["decision_rationale"], (
        f"LLM structured output failed, fell back to rule-based.\n"
        f"Rationale: {result['decision_rationale']}"
    )
    assert len(result["decision_rationale"]) > 20
