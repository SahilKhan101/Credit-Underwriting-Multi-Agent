"""
Fraud Detection Agent
Rule-based checks for velocity, cash withdrawal patterns, synthetic identity signals,
and income-application consistency. Outputs a fraud_score (0–1000 where 1000=max fraud).
The Decision Agent inverts this: fraud contribution = 1000 - fraud_score.
"""
from __future__ import annotations

import time
from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_fast_llm
from config import config


def _velocity_check(bureau: dict) -> tuple[bool, str | None]:
    enq = bureau.get("enquiries_last_6m", 0)
    if enq > config.ENQUIRY_VELOCITY_LIMIT:
        return True, f"VELOCITY FLAG: {enq} loan enquiries in last 6 months (limit {config.ENQUIRY_VELOCITY_LIMIT})"
    return False, None


def _cash_withdrawal_check(bank_stmts: list[dict]) -> tuple[bool, float, str | None]:
    if not bank_stmts:
        return False, 0.0, None
    total_credits = sum(t["amount"] for t in bank_stmts if t.get("type") == "credit")
    cash_out = sum(
        t["amount"] for t in bank_stmts
        if t.get("type") == "debit"
        and any(k in t.get("description", "").lower() for k in ("atm", "cash", "withdrawal"))
    )
    ratio = cash_out / max(total_credits, 1)
    if ratio > config.CASH_WITHDRAWAL_RATIO_LIMIT:
        return True, ratio, (
            f"HIGH CASH WITHDRAWAL: {ratio:.0%} of credits withdrawn as cash "
            f"(limit {config.CASH_WITHDRAWAL_RATIO_LIMIT:.0%})"
        )
    return False, ratio, None


def _income_consistency_check(profile: dict, bureau: dict) -> tuple[bool, str | None]:
    stated     = profile.get("monthly_income", 0)
    emp_type   = profile.get("employment_type", "SALARIED")
    total_emi  = bureau.get("total_existing_emi", 0)

    # Cross-check: if existing EMI > 80% of stated income, very suspicious
    if stated > 0 and total_emi / stated > 0.80:
        return True, (
            f"INCOME-EMI INCONSISTENCY: existing EMI ₹{total_emi:,.0f} is "
            f"{total_emi/stated:.0%} of stated income ₹{stated:,.0f}"
        )
    return False, None


def _address_consistency_check(profile: dict) -> tuple[bool, str | None]:
    city  = profile.get("city", "")
    state_ = profile.get("state", "")
    pin   = str(profile.get("pin_code", ""))

    # Known PIN prefix → state mapping (simplified)
    STATE_PIN_PREFIX = {
        "Maharashtra": ["40", "41", "42", "43", "44"],
        "Karnataka":   ["56", "57", "58"],
        "Delhi":       ["11"],
        "Telangana":   ["50"],
        "Tamil Nadu":  ["60", "62", "63", "64"],
        "Gujarat":     ["36", "38", "39"],
    }
    prefix = pin[:2]
    expected_prefixes = STATE_PIN_PREFIX.get(state_, [])
    if expected_prefixes and prefix not in expected_prefixes:
        return True, f"ADDRESS INCONSISTENCY: PIN {pin} does not match state '{state_}'"
    return False, None


def _compute_fraud_score(
    velocity_flag: bool,
    cash_flag: bool,
    income_flag: bool,
    address_flag: bool,
    cash_ratio: float,
) -> float:
    score = 0.0
    if velocity_flag:  score += 250
    if cash_flag:      score += 200 + min(100, (cash_ratio - config.CASH_WITHDRAWAL_RATIO_LIMIT) * 500)
    if income_flag:    score += 300
    if address_flag:   score += 150
    return round(min(1000.0, score), 2)


_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fraud analyst at an Indian NBFC. Write a concise 2–3 sentence "
        "fraud assessment for an underwriting file. Cite specific signals. No preamble.",
    ),
    (
        "human",
        "Fraud signals detected:\n"
        "Velocity flag: {velocity} | Cash withdrawal flag: {cash} | "
        "Income inconsistency: {income} | Address inconsistency: {addr}\n"
        "Cash withdrawal ratio: {cash_ratio:.0%} | Fraud score: {score:.0f}/1000\n"
        "Flags:\n{flags}",
    ),
])


def fraud_detection_node(state: UnderwritingState) -> dict:
    t0 = time.time()
    profile = state["applicant_profile"]
    bureau  = state["bureau_report"]
    stmts   = state.get("bank_statements", [])

    velocity_flag, v_msg     = _velocity_check(bureau)
    cash_flag, cash_ratio, c_msg = _cash_withdrawal_check(stmts)
    income_flag, i_msg       = _income_consistency_check(profile, bureau)
    addr_flag, a_msg         = _address_consistency_check(profile)

    flags = [m for m in [v_msg, c_msg, i_msg, a_msg] if m]
    fraud_score = _compute_fraud_score(velocity_flag, cash_flag, income_flag, addr_flag, cash_ratio)

    try:
        chain = _NARRATIVE_PROMPT | get_fast_llm()
        resp  = chain.invoke({
            "velocity": velocity_flag, "cash": cash_flag,
            "income": income_flag, "addr": addr_flag,
            "cash_ratio": cash_ratio, "score": fraud_score,
            "flags": "\n".join(f"• {f}" for f in flags) if flags else "• None detected",
        })
        narrative = resp.content
    except Exception as exc:
        narrative = f"Fraud analysis completed (LLM unavailable: {exc})"

    elapsed = round((time.time() - t0) * 1000)
    return {
        "fraud_signals": {
            "velocity_flag": velocity_flag,
            "cash_withdrawal_flag": cash_flag,
            "income_inconsistency_flag": income_flag,
            "address_inconsistency_flag": addr_flag,
            "cash_withdrawal_ratio": round(cash_ratio, 4),
            "fraud_score": fraud_score,         # higher = more fraudulent
            "any_fraud_flag": any([velocity_flag, cash_flag, income_flag, addr_flag]),
            "summary": narrative,
            "flags": flags,
        },
        "agent_trace": [f"[Fraud Agent] fraud_score={fraud_score:.0f}/1000 flags={len(flags)} ({elapsed}ms)"],
    }
