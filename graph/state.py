from typing import TypedDict, Annotated, List, Optional
import operator


class UnderwritingState(TypedDict, total=False):
    # ── Required inputs (must be provided at invoke time) ─────────────────
    application_id: str
    applicant_profile: dict        # name, age, employment, income, loan_ask, city …
    bureau_report: dict            # CIBIL-like JSON
    bank_statements: List[dict]    # list of 6-month transaction records

    # ── Agent outputs (populated progressively during graph execution) ─────
    bureau_analysis: dict          # scores, flags, FOIR, utilization, narrative
    income_analysis: dict          # estimated/stated income, stability, discrepancy
    bank_statement_analysis: dict  # ABB, bounce_rate, salary regularity, cashflow
    fraud_signals: dict            # velocity, cash ratio, consistency flags
    policy_check: dict             # eligible/ineligible, cited policies, violations

    # ── Decision layer ─────────────────────────────────────────────────────
    credit_score: float            # composite 0–1000
    recommendation: str            # AUTO_APPROVE | APPROVE_WITH_CONDITIONS | REFER | AUTO_REJECT
    confidence: float              # 0.0–1.0
    decision_rationale: str
    key_strengths: List[str]
    key_risks: List[str]
    conditions: List[str]          # approval conditions if any

    # ── Report ─────────────────────────────────────────────────────────────
    credit_memo: str               # full markdown credit memo

    # ── Metadata ───────────────────────────────────────────────────────────
    # Reducer fields: each agent appends; LangGraph merges during parallel fan-in
    agent_trace: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    processing_start_ms: int
    processing_time_ms: int


def initial_state(
    application_id: str,
    applicant_profile: dict,
    bureau_report: dict,
    bank_statements: List[dict],
) -> UnderwritingState:
    """Build the starting state for the graph — all agent outputs absent."""
    return UnderwritingState(
        application_id=application_id,
        applicant_profile=applicant_profile,
        bureau_report=bureau_report,
        bank_statements=bank_statements,
        agent_trace=[],
        errors=[],
    )
