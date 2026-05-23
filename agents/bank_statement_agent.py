"""
Bank Statement Agent
Analyses 6-month transaction records using Pandas:
  - Average Bank Balance (ABB) vs EMI
  - Salary regularity
  - Bounce rate
  - Cash withdrawal ratio
  - Net monthly cash-flow trend
Outputs a normalised cashflow score (0–1000).
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from graph.state import UnderwritingState
from utils.llm import get_fast_llm
from config import config


def _build_df(statements: list[dict]) -> pd.DataFrame:
    if not statements:
        return pd.DataFrame(columns=["date", "description", "amount", "type", "balance_after"])
    df = pd.DataFrame(statements)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["balance_after"] = pd.to_numeric(df.get("balance_after", 0), errors="coerce").fillna(0)
    return df


def _extract_features(df: pd.DataFrame, monthly_income: float, requested_emi: float) -> dict:
    if df.empty:
        return {
            "avg_monthly_balance": 0, "bounce_rate": 0, "salary_regularity_score": 0,
            "abb_to_emi_ratio": 0, "cash_withdrawal_ratio": 0, "net_monthly_cashflow": 0,
            "months_analysed": 0, "total_credits": 0, "total_debits": 0,
        }

    credits = df[df["type"] == "credit"]
    debits  = df[df["type"] == "debit"]
    bounces = df[df.get("bounced", False) == True] if "bounced" in df.columns else pd.DataFrame()

    total_credits = float(credits["amount"].sum())
    total_debits  = float(debits["amount"].sum())

    # ABB — use balance snapshots if available, else approximate
    avg_balance = float(df["balance_after"].mean()) if "balance_after" in df.columns else 0

    # Bounce rate
    bounce_rate = len(bounces) / max(len(debits), 1)

    # Salary credits — months with salary
    salary_txns = credits[credits["description"].str.lower().str.contains("salary|sal cr|payroll", na=False)]
    salary_months_count = salary_txns.groupby(salary_txns["date"].dt.to_period("M")).size()
    months_analysed = df["date"].dt.to_period("M").nunique()
    salary_regularity = len(salary_months_count) / max(months_analysed, 1)

    # Cash withdrawals
    cash_txns = debits[debits["description"].str.lower().str.contains("atm|cash|withdrawal", na=False)]
    cash_out  = float(cash_txns["amount"].sum())
    cash_ratio = cash_out / max(total_credits, 1)

    # Net cashflow per month
    net_monthly = (total_credits - total_debits) / max(months_analysed, 1)

    # ABB-to-EMI ratio
    abb_emi_ratio = avg_balance / max(requested_emi, 1)

    return {
        "avg_monthly_balance": round(avg_balance, 2),
        "bounce_rate": round(bounce_rate, 4),
        "salary_regularity_score": round(salary_regularity, 4),
        "abb_to_emi_ratio": round(abb_emi_ratio, 4),
        "cash_withdrawal_ratio": round(cash_ratio, 4),
        "net_monthly_cashflow": round(net_monthly, 2),
        "months_analysed": months_analysed,
        "total_credits": round(total_credits, 2),
        "total_debits": round(total_debits, 2),
    }


def _compute_cashflow_score(feats: dict, requested_emi: float) -> tuple[float, list[str]]:
    flags: list[str] = []

    bounce    = feats["bounce_rate"]
    cash_rat  = feats["cash_withdrawal_ratio"]
    regularity = feats["salary_regularity_score"]
    abb_emi   = feats["abb_to_emi_ratio"]
    net_cf    = feats["net_monthly_cashflow"]

    if bounce > 0.05:
        flags.append(f"HIGH BOUNCE RATE: {bounce:.0%} of debit transactions")
    if cash_rat > config.CASH_WITHDRAWAL_RATIO_LIMIT:
        flags.append(f"HIGH CASH WITHDRAWAL: {cash_rat:.0%} of total credits")
    if regularity < 0.7:
        flags.append(f"IRREGULAR SALARY CREDITS: regularity {regularity:.0%}")
    if abb_emi < 1.5:
        flags.append(f"LOW ABB/EMI RATIO: {abb_emi:.2f}× (target ≥1.5×)")
    if net_cf < 0:
        flags.append(f"NEGATIVE NET CASHFLOW: ₹{net_cf:,.0f}/month")

    # Score components (each out of partial budget)
    bounce_score    = max(0.0, (1 - bounce / 0.30)) * 200
    cash_score      = max(0.0, (1 - cash_rat / 0.60)) * 200
    regularity_score = regularity * 250
    abb_score       = min(200.0, abb_emi * 50)
    cf_score        = 150.0 if net_cf >= 0 else max(0.0, 150 + net_cf / 1000)

    score = max(0.0, min(1000.0, bounce_score + cash_score + regularity_score + abb_score + cf_score))
    return round(score, 2), flags


_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a credit analyst reviewing bank statements for an Indian NBFC. "
        "Write a concise 2–3 sentence cashflow assessment. Cite numbers. No preamble.",
    ),
    (
        "human",
        "Bank statement summary (6 months):\n"
        "Avg monthly balance: ₹{abb:,.0f} | EMI requested: ₹{emi:,.0f} | "
        "ABB/EMI ratio: {abb_emi:.2f}× | Bounce rate: {bounce:.0%} | "
        "Salary regularity: {reg:.0%} | Cash withdrawal ratio: {cash:.0%} | "
        "Net monthly cashflow: ₹{cf:,.0f}\n"
        "Cashflow score: {score:.0f}/1000\nFlags: {flags}",
    ),
])


def bank_statement_node(state: UnderwritingState) -> dict:
    t0 = time.time()
    stmts   = state.get("bank_statements", [])
    profile = state["applicant_profile"]

    monthly_income = float(profile.get("monthly_income", 0))
    requested_emi  = float(profile.get("requested_emi", 1))

    df    = _build_df(stmts)
    feats = _extract_features(df, monthly_income, requested_emi)
    score, flags = _compute_cashflow_score(feats, requested_emi)

    try:
        chain = _NARRATIVE_PROMPT | get_fast_llm()
        resp  = chain.invoke({
            "abb": feats["avg_monthly_balance"], "emi": requested_emi,
            "abb_emi": feats["abb_to_emi_ratio"], "bounce": feats["bounce_rate"],
            "reg": feats["salary_regularity_score"], "cash": feats["cash_withdrawal_ratio"],
            "cf": feats["net_monthly_cashflow"], "score": score,
            "flags": "; ".join(flags) if flags else "None",
        })
        narrative = resp.content
    except Exception as exc:
        narrative = f"Bank statement analysis completed (LLM unavailable: {exc})"

    elapsed = round((time.time() - t0) * 1000)
    return {
        "bank_statement_analysis": {**feats, "cashflow_score": score, "summary": narrative, "flags": flags},
        "agent_trace": [f"[Bank Stmt Agent] score={score:.0f}/1000 flags={len(flags)} ({elapsed}ms)"],
    }
