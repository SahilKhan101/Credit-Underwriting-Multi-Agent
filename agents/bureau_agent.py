"""
Bureau Analysis Agent
Parses the CIBIL-like bureau report, derives FOIR/utilization/DPD features,
computes a normalized bureau score (0–1000), and generates an LLM narrative.
"""
from __future__ import annotations

import time
from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_fast_llm
from config import config


def _compute_bureau_score(
    cibil_score: int,
    utilization: float,
    dpd_30: int,
    dpd_90: int,
    written_off: bool,
    enquiries_6m: int,
    oldest_months: int,
    foir: float,
) -> tuple[float, bool, list[str]]:
    flags: list[str] = []
    derogatory = False

    # ── Derogatory checks ──────────────────────────────────────────────────
    if dpd_90 > 0 or written_off:
        derogatory = True
        flags.append("DEROGATORY: DPD-90+ or written-off account detected")
    if dpd_30 > 2:
        flags.append(f"HIGH DPD-30: {dpd_30} instances in last 12 months")
    if utilization > 0.75:
        flags.append(f"HIGH UTILIZATION: {utilization:.0%} (>75%)")
    if enquiries_6m > config.ENQUIRY_VELOCITY_LIMIT:
        flags.append(f"EXCESSIVE ENQUIRIES: {enquiries_6m} in last 6 months")
    if foir > config.MAX_FOIR:
        flags.append(f"HIGH FOIR: {foir:.0%} exceeds {config.MAX_FOIR:.0%} limit")
    if cibil_score < config.MIN_CIBIL:
        flags.append(f"LOW CIBIL: {cibil_score} below minimum {config.MIN_CIBIL}")

    # ── Scoring components ─────────────────────────────────────────────────
    # CIBIL base (300–900 → 0–400 pts)
    cibil_pts = max(0.0, (cibil_score - 300) / 600) * 400

    # Age-of-credit bonus (max 50 pts)
    age_pts = min(50.0, oldest_months / 2.0)

    # Penalties
    dpd_pen = min(200.0, dpd_30 * 20 + dpd_90 * 80 + (150 if written_off else 0))
    util_pen = max(0.0, (utilization - 0.30) / 0.70) * 100
    enq_pen  = min(50.0, max(0, enquiries_6m - 2) * 15)
    foir_pen = max(0.0, (foir - 0.40) / 0.60) * 100 if foir > 0.40 else 0.0

    # Remaining budget fills out to 400 pts for non-CIBIL quality
    quality_pts = max(0.0, 400 - dpd_pen - util_pen - enq_pen - foir_pen) + age_pts

    score = max(0.0, min(1000.0, cibil_pts + quality_pts))

    # Derogatory hard cap
    if derogatory:
        score = min(score, 350.0)

    return score, derogatory, flags


_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior credit analyst at an Indian NBFC. "
        "Write a concise 2–3 sentence bureau assessment for an underwriting file. "
        "Be specific: cite numbers, flag risks or strengths. No preamble.",
    ),
    (
        "human",
        "Bureau snapshot:\n"
        "CIBIL: {cibil} | Utilization: {util:.0%} | DPD-30 (12m): {dpd30} | "
        "DPD-90 (12m): {dpd90} | Enquiries (6m): {enq} | "
        "Oldest account: {age}m | Written-off: {wo} | FOIR: {foir:.0%}\n"
        "Bureau score: {score:.0f}/1000\n"
        "Flags: {flags}",
    ),
])


def bureau_analysis_node(state: UnderwritingState) -> dict:
    t0 = time.time()
    bureau  = state["bureau_report"]
    profile = state["applicant_profile"]

    cibil       = bureau.get("cibil_score", 600)
    utilization = bureau.get("credit_utilization", 0.5)
    dpd_30      = bureau.get("dpd_30_count", 0)
    dpd_90      = bureau.get("dpd_90_count", 0)
    written_off = bureau.get("written_off_flag", False)
    enquiries   = bureau.get("enquiries_last_6m", 0)
    oldest      = bureau.get("oldest_account_months", 12)
    existing_emi = bureau.get("total_existing_emi", 0.0)

    monthly_income = profile.get("monthly_income", 1)
    requested_emi  = profile.get("requested_emi", 0)
    foir = (existing_emi + requested_emi) / max(monthly_income, 1)

    score, derogatory, flags = _compute_bureau_score(
        cibil, utilization, dpd_30, dpd_90, written_off, enquiries, oldest, foir
    )

    try:
        chain = _NARRATIVE_PROMPT | get_fast_llm()
        resp  = chain.invoke({
            "cibil": cibil, "util": utilization, "dpd30": dpd_30,
            "dpd90": dpd_90, "enq": enquiries, "age": oldest,
            "wo": written_off, "foir": foir, "score": score,
            "flags": "; ".join(flags) if flags else "None",
        })
        narrative = resp.content
    except Exception as exc:
        narrative = f"Bureau analysis completed (LLM unavailable: {exc})"

    elapsed = round((time.time() - t0) * 1000)
    return {
        "bureau_analysis": {
            "cibil_score": cibil,
            "credit_utilization": utilization,
            "dpd_30_count": dpd_30,
            "dpd_90_count": dpd_90,
            "derogatory_flag": derogatory,
            "foir": round(foir, 4),
            "oldest_account_months": oldest,
            "enquiries_last_6m": enquiries,
            "existing_emi": existing_emi,
            "bureau_score": round(score, 2),
            "summary": narrative,
            "flags": flags,
        },
        "agent_trace": [f"[Bureau Agent] score={score:.0f}/1000 flags={len(flags)} ({elapsed}ms)"],
    }
