"""
Report Generation Agent
Produces a structured Markdown credit memo — the final artefact handed to
the credit committee or stored in the audit trail.
Uses Claude Sonnet for the narrative; wraps it in a deterministic template.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_smart_llm


_REC_LABELS = {
    "AUTO_APPROVE":              ("✅ AUTO APPROVE", "The application meets all policy criteria and scores above the auto-approval threshold."),
    "APPROVE_WITH_CONDITIONS":   ("🟡 APPROVE WITH CONDITIONS", "The application is broadly acceptable but requires specific conditions before disbursement."),
    "REFER_TO_CREDIT_COMMITTEE": ("🔵 REFER TO CREDIT COMMITTEE", "Borderline case requiring senior credit officer review."),
    "AUTO_REJECT":               ("❌ AUTO REJECT", "The application does not meet minimum eligibility criteria."),
}

_MEMO_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a credit analyst writing a formal credit assessment memo for an Indian NBFC. "
        "Write a focused 3–4 paragraph narrative covering: (1) applicant profile & loan request, "
        "(2) credit risk analysis, (3) income & cashflow assessment, (4) recommendation rationale. "
        "Use professional banking language. Cite specific numbers. No markdown headers — plain paragraphs only.",
    ),
    (
        "human",
        """Application ID: {app_id}
Applicant: {name} | {emp} | Age {age} | ₹{income:,.0f}/month | {city}
Loan: ₹{loan:,.0f} for {tenure} months ({purpose})

Credit Score: {score:.0f}/1000 | Recommendation: {rec}
Confidence: {conf:.0%}

Key Strengths: {strengths}
Key Risks: {risks}
Conditions: {conditions}

Agent Summaries:
[Bureau] {bureau_sum}
[Income] {income_sum}
[Cashflow] {cashflow_sum}
[Fraud] {fraud_sum}
[Policy] {policy_sum}

Write the credit assessment narrative.""",
    ),
])


def _score_breakdown_table(
    bureau_score: float,
    income_score: float,
    cashflow_score: float,
    fraud_score: float,
    policy_score: float,
    total: float,
) -> str:
    fraud_contrib = 1000.0 - fraud_score
    rows = [
        ("Bureau Analysis",    bureau_score,   0.35),
        ("Income Verification", income_score,  0.25),
        ("Bank Statement / Cashflow", cashflow_score, 0.20),
        ("Fraud Assessment (inverted)", fraud_contrib, 0.15),
        ("Policy Compliance",  policy_score,  0.05),
    ]
    lines = ["| Component | Raw Score | Weight | Contribution |",
             "|---|---|---|---|"]
    for name, raw, w in rows:
        lines.append(f"| {name} | {raw:.0f}/1000 | {w:.0%} | {raw*w:.1f} |")
    lines.append(f"| **COMPOSITE SCORE** | | | **{total:.0f}/1000** |")
    return "\n".join(lines)


def report_generation_node(state: UnderwritingState) -> dict:
    t0 = time.time()

    profile  = state["applicant_profile"]
    bureau_a = state.get("bureau_analysis", {})
    income_a = state.get("income_analysis", {})
    bank_a   = state.get("bank_statement_analysis", {})
    fraud_a  = state.get("fraud_signals", {})
    policy_a = state.get("policy_check", {})

    rec       = state.get("recommendation", "REFER_TO_CREDIT_COMMITTEE")
    score     = float(state.get("credit_score", 0))
    conf      = float(state.get("confidence", 0))
    rationale = state.get("decision_rationale", "")
    strengths = state.get("key_strengths", [])
    risks     = state.get("key_risks", [])
    conditions = state.get("conditions", [])

    rec_label, rec_desc = _REC_LABELS.get(rec, ("❓ UNKNOWN", ""))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Score breakdown table
    score_table = _score_breakdown_table(
        float(bureau_a.get("bureau_score", 500)),
        float(income_a.get("income_score", 500)),
        float(bank_a.get("cashflow_score", 500)),
        float(fraud_a.get("fraud_score", 0)),
        float(policy_a.get("policy_score", 800)),
        score,
    )

    try:
        chain = _MEMO_PROMPT | get_smart_llm()
        resp  = chain.invoke({
            "app_id":   state["application_id"],
            "name":     profile.get("name", "Applicant"),
            "emp":      profile.get("employment_type", "N/A"),
            "age":      profile.get("age", 0),
            "income":   profile.get("monthly_income", 0),
            "city":     profile.get("city", "N/A"),
            "loan":     profile.get("loan_amount", 0),
            "tenure":   profile.get("loan_tenure_months", 0),
            "purpose":  profile.get("loan_purpose", "N/A"),
            "score":    score, "rec": rec, "conf": conf,
            "strengths": "; ".join(strengths) or "N/A",
            "risks":    "; ".join(risks) or "None flagged",
            "conditions": "; ".join(conditions) or "None",
            "bureau_sum":   bureau_a.get("summary", ""),
            "income_sum":   income_a.get("summary", ""),
            "cashflow_sum": bank_a.get("summary", ""),
            "fraud_sum":    fraud_a.get("summary", ""),
            "policy_sum":   policy_a.get("summary", ""),
        })
        narrative = resp.content
    except Exception as exc:
        err_str = str(exc)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            reason = "LLM daily quota exhausted — narrative unavailable. Rule-based decision is still valid."
        else:
            reason = f"LLM unavailable: {err_str[:120]}"
        narrative = f"*{reason}*\n\nDecision: **{rec}** | Score: **{score:.0f}/1000**"

    # Assemble the full memo
    all_flags = []
    for src, data in [("Bureau", bureau_a), ("Income", income_a),
                      ("Cashflow", bank_a), ("Fraud", fraud_a), ("Policy", policy_a)]:
        for f in data.get("flags", []):
            all_flags.append(f"**[{src}]** {f}")

    memo = f"""# Credit Assessment Memo
**Application ID:** {state["application_id"]}
**Date:** {now}
**Applicant:** {profile.get("name", "N/A")} | Age {profile.get("age", 0)} | {profile.get("employment_type", "N/A")}
**Loan Request:** ₹{profile.get("loan_amount", 0):,.0f} for {profile.get("loan_tenure_months", 0)} months ({profile.get("loan_purpose", "N/A")})

---

## Decision: {rec_label}

> {rec_desc}

**Composite Credit Score: {score:.0f} / 1000**
**Confidence: {conf:.0%}**

---

## Score Breakdown

{score_table}

---

## Underwriting Narrative

{narrative}

---

## Key Decision Factors

**Strengths:**
{chr(10).join("- " + s for s in strengths) if strengths else "- (none identified)"}

**Risks & Concerns:**
{chr(10).join("- " + r for r in risks) if risks else "- None flagged"}

{"**Approval Conditions:**" + chr(10) + chr(10).join("- " + c for c in conditions) if conditions else ""}

---

## All Flags Raised During Underwriting

{chr(10).join("- " + f for f in all_flags) if all_flags else "- No flags raised"}

---

## Agent Audit Trail

{chr(10).join("- " + t for t in state.get("agent_trace", []))}

---
*Generated by CreditIQ Multi-Agent Underwriting System*
"""

    elapsed = round((time.time() - t0) * 1000)
    return {
        "credit_memo": memo,
        "agent_trace": [f"[Report Agent] memo generated ({elapsed}ms)"],
    }
