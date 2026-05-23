"""
Decision Agent
Aggregates the five analysis dicts, computes the composite credit score (0–1000)
using configured weights, then calls Claude Sonnet with Pydantic structured output
for the final recommendation and rationale.
"""
from __future__ import annotations

import time
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_smart_llm
from config import config


class CreditDecision(BaseModel):
    recommendation: Literal[
        "AUTO_APPROVE",
        "APPROVE_WITH_CONDITIONS",
        "REFER_TO_CREDIT_COMMITTEE",
        "AUTO_REJECT",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_strengths: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(
        default_factory=list,
        description="Approval conditions, empty unless APPROVE_WITH_CONDITIONS",
    )
    decision_rationale: str


def _composite_score(
    bureau_score: float,
    income_score: float,
    cashflow_score: float,
    fraud_score: float,   # higher = more fraudulent; inverted here
    policy_score: float,
) -> float:
    fraud_contribution = 1000.0 - fraud_score   # invert: low fraud → high contribution
    raw = (
        bureau_score    * config.BUREAU_WEIGHT
        + income_score  * config.INCOME_WEIGHT
        + cashflow_score * config.CASHFLOW_WEIGHT
        + fraud_contribution * config.FRAUD_WEIGHT
        + policy_score  * config.POLICY_WEIGHT
    )
    return round(max(0.0, min(1000.0, raw)), 2)


def _score_to_recommendation(score: float, policy_eligible: bool) -> str:
    if not policy_eligible:
        return "AUTO_REJECT"
    if score >= config.AUTO_APPROVE_THRESHOLD:
        return "AUTO_APPROVE"
    if score >= config.APPROVE_THRESHOLD:
        return "APPROVE_WITH_CONDITIONS"
    if score >= config.REFER_THRESHOLD:
        return "REFER_TO_CREDIT_COMMITTEE"
    return "AUTO_REJECT"


_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the Chief Credit Officer of an Indian NBFC. "
        "Given the underwriting summary below, provide the credit decision in valid JSON "
        "matching the CreditDecision schema. Be precise, cite numbers, and be concise.",
    ),
    (
        "human",
        """Application: {app_id}
Applicant: {name} | Age: {age} | Employment: {emp} | Monthly Income: ₹{income:,.0f}
Loan Ask: ₹{loan:,.0f} | Tenure: {tenure} months | Purpose: {purpose}

── Agent Scores ──────────────────────────────
Bureau Score:   {bureau_score:.0f}/1000  | {bureau_summary}
Income Score:   {income_score:.0f}/1000  | {income_summary}
Cashflow Score: {cashflow_score:.0f}/1000 | {cashflow_summary}
Fraud Score:    {fraud_score:.0f}/1000 (inverted contribution) | {fraud_summary}
Policy Score:   {policy_score:.0f}/1000  | {policy_summary}

── Composite Credit Score ────────────────────
SCORE: {credit_score:.0f}/1000
Rule-based recommendation: {rule_rec}

── All Flags ─────────────────────────────────
{all_flags}

Provide your structured decision.""",
    ),
])


def decision_agent_node(state: UnderwritingState) -> dict:
    t0 = time.time()

    bureau_a   = state.get("bureau_analysis", {})
    income_a   = state.get("income_analysis", {})
    bank_a     = state.get("bank_statement_analysis", {})
    fraud_a    = state.get("fraud_signals", {})
    policy_a   = state.get("policy_check", {})
    profile    = state["applicant_profile"]

    bureau_score   = float(bureau_a.get("bureau_score", 500))
    income_score   = float(income_a.get("income_score", 500))
    cashflow_score = float(bank_a.get("cashflow_score", 500))
    fraud_score    = float(fraud_a.get("fraud_score", 0))
    policy_score   = float(policy_a.get("policy_score", 800))
    policy_elig    = bool(policy_a.get("eligible", True))

    credit_score = _composite_score(
        bureau_score, income_score, cashflow_score, fraud_score, policy_score
    )
    rule_rec = _score_to_recommendation(credit_score, policy_elig)

    all_flags = []
    for src, data in [("Bureau", bureau_a), ("Income", income_a),
                      ("Cashflow", bank_a), ("Fraud", fraud_a), ("Policy", policy_a)]:
        for f in data.get("flags", []):
            all_flags.append(f"[{src}] {f}")

    try:
        # method="function_calling" avoids the thinking-token newline corruption
        # that occurs with JSON mode on gemini-2.5-flash
        llm = get_smart_llm().with_structured_output(CreditDecision, method="function_calling")
        decision: CreditDecision = llm.invoke(_DECISION_PROMPT.format_messages(
            app_id        = state["application_id"],
            name          = profile.get("name", "Applicant"),
            age           = profile.get("age", 0),
            emp           = profile.get("employment_type", "N/A"),
            income        = profile.get("monthly_income", 0),
            loan          = profile.get("loan_amount", 0),
            tenure        = profile.get("loan_tenure_months", 0),
            purpose       = profile.get("loan_purpose", "N/A"),
            bureau_score  = bureau_score,
            income_score  = income_score,
            cashflow_score = cashflow_score,
            fraud_score   = fraud_score,
            policy_score  = policy_score,
            credit_score  = credit_score,
            rule_rec      = rule_rec,
            bureau_summary  = bureau_a.get("summary", ""),
            income_summary  = income_a.get("summary", ""),
            cashflow_summary = bank_a.get("summary", ""),
            fraud_summary   = fraud_a.get("summary", ""),
            policy_summary  = policy_a.get("summary", ""),
            all_flags     = "\n".join(f"• {f}" for f in all_flags) if all_flags else "• None",
        ))
        rec      = decision.recommendation
        conf     = decision.confidence
        rationale = decision.decision_rationale
        strengths = decision.key_strengths
        risks     = decision.key_risks
        conditions = decision.conditions

    except Exception as exc:
        # Fallback to rule-based decision
        rec       = rule_rec
        conf      = 0.75
        rationale = f"Rule-based decision (LLM unavailable: {exc})"
        strengths = []
        risks     = all_flags[:3]
        conditions = []

    elapsed = round((time.time() - t0) * 1000)
    return {
        "credit_score": credit_score,
        "recommendation": rec,
        "confidence": conf,
        "decision_rationale": rationale,
        "key_strengths": strengths,
        "key_risks": risks,
        "conditions": conditions,
        "agent_trace": [
            f"[Decision Agent] score={credit_score:.0f} rec={rec} conf={conf:.0%} ({elapsed}ms)"
        ],
    }
