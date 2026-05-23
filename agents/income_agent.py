"""
Income Verification Agent
Estimates income from applicant profile (rule-based proxy for XGBoost model),
compares against stated income, derives stability score from bank statements,
and outputs a normalized income score (0–1000).
"""
from __future__ import annotations

import time
import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_fast_llm
from config import config

# Employment multiplier table (mimics XGBoost learned weights)
_EMPLOYMENT_MULTIPLIERS = {
    "SALARIED":       1.00,
    "SELF_EMPLOYED":  0.85,   # income haircut due to higher variability
    "BUSINESS":       0.80,
    "PROFESSIONAL":   1.10,   # doctors, CAs, engineers — premium
    "GOVERNMENT":     1.05,
}

_CITY_TIER = {
    "Mumbai": 1.3, "Delhi": 1.25, "Bangalore": 1.25, "Hyderabad": 1.15,
    "Chennai": 1.15, "Pune": 1.10, "Kolkata": 1.05, "Ahmedabad": 1.05,
}


def _estimate_income(profile: dict) -> float:
    """Rule-based income estimator (plug-in point for XGBoost model)."""
    stated  = profile.get("monthly_income", 0)
    emp     = profile.get("employment_type", "SALARIED")
    city    = profile.get("city", "")
    age     = profile.get("age", 30)

    multiplier = _EMPLOYMENT_MULTIPLIERS.get(emp, 1.0)
    city_adj   = _CITY_TIER.get(city, 1.0)
    age_adj    = 1.0 + max(0, (age - 25) * 0.008)   # experience premium up to 5%

    # Estimated income is stated income adjusted by model factors
    return stated * multiplier * city_adj * age_adj


def _salary_stability(bank_statements: list[dict]) -> tuple[float, float]:
    """Return (cv, regularity_score) from bank statement salary credits."""
    salary_credits = [
        txn["amount"]
        for txn in bank_statements
        if txn.get("type") == "credit" and "salary" in txn.get("description", "").lower()
    ]
    if len(salary_credits) < 2:
        return 0.5, 0.5   # insufficient data — neutral

    arr = np.array(salary_credits, dtype=float)
    cv  = float(arr.std() / arr.mean()) if arr.mean() > 0 else 1.0
    regularity = max(0.0, 1.0 - cv * 2)   # cv=0 → 1.0, cv=0.5+ → 0.0
    return round(cv, 4), round(regularity, 4)


def _compute_income_score(
    stated: float,
    estimated: float,
    loan_amount: float,
    stability: float,
    discrepancy_pct: float,
) -> tuple[float, list[str]]:
    flags: list[str] = []

    if discrepancy_pct > config.INCOME_DISCREPANCY_PCT_LIMIT:
        flags.append(
            f"INCOME DISCREPANCY: {discrepancy_pct:.0%} gap between "
            f"stated (₹{stated:,.0f}) and estimated (₹{estimated:,.0f})"
        )

    annual_income    = stated * 12
    income_to_loan   = annual_income / max(loan_amount, 1)
    if income_to_loan < 3.0:
        flags.append(f"LOW INCOME-LOAN RATIO: {income_to_loan:.1f}× (min 3×)")
    if stability < 0.4:
        flags.append(f"LOW INCOME STABILITY: regularity score {stability:.2f}")

    # Score components
    adequacy_score = min(400.0, max(0.0, (income_to_loan / config.MAX_LOAN_TO_INCOME_RATIO) * 400))
    stability_score = stability * 300
    discrepancy_pen = min(200.0, discrepancy_pct * 500) if discrepancy_pct > 0.20 else 0.0

    score = max(0.0, min(1000.0, adequacy_score + stability_score + 300 - discrepancy_pen))
    return round(score, 2), flags


_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a credit analyst. Write a concise 2–3 sentence income assessment "
        "for an Indian NBFC underwriting file. Cite specific numbers. No preamble.",
    ),
    (
        "human",
        "Income data:\n"
        "Employment: {emp} | Stated monthly income: ₹{stated:,.0f} | "
        "Estimated income: ₹{estimated:,.0f} | Discrepancy: {disc:.0%}\n"
        "Loan ask: ₹{loan:,.0f} | Income-to-loan ratio: {ratio:.1f}× | "
        "Income stability (regularity): {stab:.2f}\n"
        "Income score: {score:.0f}/1000\nFlags: {flags}",
    ),
])


def income_verification_node(state: UnderwritingState) -> dict:
    t0 = time.time()
    profile = state["applicant_profile"]
    stmts   = state.get("bank_statements", [])

    stated    = float(profile.get("monthly_income", 0))
    loan_amt  = float(profile.get("loan_amount", 1))
    emp_type  = profile.get("employment_type", "SALARIED")

    estimated      = _estimate_income(profile)
    discrepancy    = abs(stated - estimated) / max(estimated, 1)
    cv, stability  = _salary_stability(stmts)
    income_to_loan = (stated * 12) / max(loan_amt, 1)

    score, flags = _compute_income_score(
        stated, estimated, loan_amt, stability, discrepancy
    )

    try:
        chain = _NARRATIVE_PROMPT | get_fast_llm()
        resp  = chain.invoke({
            "emp": emp_type, "stated": stated, "estimated": estimated,
            "disc": discrepancy, "loan": loan_amt, "ratio": income_to_loan,
            "stab": stability, "score": score,
            "flags": "; ".join(flags) if flags else "None",
        })
        narrative = resp.content
    except Exception as exc:
        narrative = f"Income analysis completed (LLM unavailable: {exc})"

    elapsed = round((time.time() - t0) * 1000)
    return {
        "income_analysis": {
            "stated_income": stated,
            "estimated_income": round(estimated, 2),
            "discrepancy_pct": round(discrepancy, 4),
            "discrepancy_flag": discrepancy > config.INCOME_DISCREPANCY_PCT_LIMIT,
            "income_stability_cv": cv,
            "income_stability_score": stability,
            "income_to_loan_ratio": round(income_to_loan, 2),
            "employment_type": emp_type,
            "income_score": score,
            "summary": narrative,
            "flags": flags,
        },
        "agent_trace": [f"[Income Agent] score={score:.0f}/1000 flags={len(flags)} ({elapsed}ms)"],
    }
