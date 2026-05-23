"""
CreditIQ — Streamlit Demo UI

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import initial_state
from graph.workflow import app as workflow

st.set_page_config(
    page_title="CreditIQ — AI Underwriting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 CreditIQ")
    st.caption("Multi-Agent Credit Underwriting System")
    st.divider()

    st.subheader("Load Sample Application")
    load_sample = st.button("Load Random Synthetic Application", use_container_width=True)
    load_approve = st.button("Load Prime Applicant (APPROVE)", use_container_width=True)
    load_reject  = st.button("Load Subprime Applicant (REJECT)", use_container_width=True)

    st.divider()
    st.caption("Stack: LangGraph · LangChain · Claude · ChromaDB")


# ── Sample data helpers ───────────────────────────────────────────────────────

def _sample_prime() -> dict:
    return {
        "profile": {
            "name": "Priya Sharma", "age": 32,
            "employment_type": "SALARIED", "monthly_income": 95000,
            "employer": "Infosys Ltd", "city": "Bangalore",
            "state": "Karnataka", "pin_code": "560001",
            "loan_amount": 500000, "loan_tenure_months": 36,
            "loan_purpose": "HOME_RENOVATION", "requested_emi": 16200,
        },
        "bureau": {
            "cibil_score": 778, "total_accounts": 4, "active_accounts": 2,
            "credit_utilization": 0.28, "dpd_30_count": 0, "dpd_90_count": 0,
            "enquiries_last_6m": 1, "oldest_account_months": 60,
            "total_existing_emi": 12000, "written_off_flag": False,
            "secured_unsecured_ratio": 0.6,
        },
    }


def _sample_reject() -> dict:
    return {
        "profile": {
            "name": "Ramesh Kumar", "age": 44,
            "employment_type": "SELF_EMPLOYED", "monthly_income": 28000,
            "employer": "Self", "city": "Nagpur",
            "state": "Maharashtra", "pin_code": "440001",
            "loan_amount": 400000, "loan_tenure_months": 48,
            "loan_purpose": "PERSONAL", "requested_emi": 11500,
        },
        "bureau": {
            "cibil_score": 548, "total_accounts": 6, "active_accounts": 4,
            "credit_utilization": 0.87, "dpd_30_count": 4, "dpd_90_count": 1,
            "enquiries_last_6m": 7, "oldest_account_months": 18,
            "total_existing_emi": 18000, "written_off_flag": True,
            "secured_unsecured_ratio": 0.2,
        },
    }


def _load_random_synthetic() -> dict | None:
    files = list(Path("data/synthetic").glob("*.json"))
    if not files:
        return None
    import random
    rec = json.loads(random.choice(files).read_text())
    return {"profile": rec["applicant_profile"], "bureau": rec["bureau_report"]}


def _gen_statements(profile: dict, n_months: int = 6) -> list[dict]:
    """Simple synthetic statement generator for UI demo."""
    import random
    from datetime import datetime, timedelta
    stmts = []
    base  = datetime.today()
    for m in range(n_months):
        dt = (base - timedelta(days=(n_months - m) * 30)).strftime("%Y-%m-%d")
        stmts.append({
            "date": dt, "description": "SALARY CR NEFT",
            "amount": profile["monthly_income"] * random.uniform(0.97, 1.03),
            "type": "credit", "balance_after": profile["monthly_income"] * 2,
        })
        stmts.append({
            "date": dt, "description": "EMI DEBIT",
            "amount": profile["requested_emi"],
            "type": "debit", "balance_after": profile["monthly_income"] * 1.5,
        })
    return stmts


# ── Session state ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "result": None, "running": False,
        "profile": _sample_prime()["profile"],
        "bureau":  _sample_prime()["bureau"],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

if load_approve:
    d = _sample_prime()
    st.session_state["profile"] = d["profile"]
    st.session_state["bureau"]  = d["bureau"]
    st.session_state["result"]  = None
elif load_reject:
    d = _sample_reject()
    st.session_state["profile"] = d["profile"]
    st.session_state["bureau"]  = d["bureau"]
    st.session_state["result"]  = None
elif load_sample:
    d = _load_random_synthetic()
    if d:
        st.session_state["profile"] = d["profile"]
        st.session_state["bureau"]  = d["bureau"]
        st.session_state["result"]  = None
    else:
        st.sidebar.warning("No synthetic data found. Run `python scripts/generate_synthetic_data.py` first.")


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("🏦 CreditIQ — AI Credit Underwriting")
st.caption("LangGraph multi-agent system: Bureau · Income · Cashflow · Fraud · Policy → Decision")

col_form, col_trace = st.columns([1, 1], gap="large")

# ── LEFT: Application form ────────────────────────────────────────────────────

with col_form:
    st.subheader("Loan Application")
    p = st.session_state["profile"]
    b = st.session_state["bureau"]

    with st.form("application_form"):
        st.markdown("**Applicant Details**")
        c1, c2 = st.columns(2)
        name    = c1.text_input("Name",  value=p.get("name", ""))
        age     = c2.number_input("Age", value=int(p.get("age", 30)), min_value=18, max_value=70)
        emp     = st.selectbox("Employment Type",
                    ["SALARIED", "SELF_EMPLOYED", "BUSINESS", "PROFESSIONAL", "GOVERNMENT"],
                    index=["SALARIED", "SELF_EMPLOYED", "BUSINESS", "PROFESSIONAL", "GOVERNMENT"].index(
                        p.get("employment_type", "SALARIED")))
        income  = st.number_input("Monthly Income (₹)", value=int(p.get("monthly_income", 50000)), min_value=10000, step=5000)
        city    = st.text_input("City", value=p.get("city", "Mumbai"))
        state_  = st.text_input("State", value=p.get("state", "Maharashtra"))

        st.markdown("**Loan Request**")
        c1, c2 = st.columns(2)
        loan    = c1.number_input("Loan Amount (₹)", value=int(p.get("loan_amount", 500000)), min_value=10000, step=10000)
        tenure  = c2.selectbox("Tenure (months)", [12, 24, 36, 48, 60, 72, 84],
                    index=[12, 24, 36, 48, 60, 72, 84].index(p.get("loan_tenure_months", 36)))
        purpose = st.selectbox("Purpose", ["HOME_RENOVATION", "EDUCATION", "MEDICAL", "VEHICLE", "PERSONAL", "BUSINESS"],
                    index=["HOME_RENOVATION", "EDUCATION", "MEDICAL", "VEHICLE", "PERSONAL", "BUSINESS"].index(
                        p.get("loan_purpose", "PERSONAL")))

        # Rough EMI calculation
        r  = 0.015
        emi_calc = int(loan * r / (1 - (1 + r) ** -tenure))
        emi = st.number_input("Monthly EMI (₹)", value=emi_calc, min_value=500, step=100)

        st.markdown("**Bureau / CIBIL**")
        c1, c2 = st.columns(2)
        cibil   = c1.slider("CIBIL Score", 300, 900, int(b.get("cibil_score", 720)))
        util    = c2.slider("Credit Utilization", 0.0, 1.0, float(b.get("credit_utilization", 0.35)))
        c1, c2, c3 = st.columns(3)
        dpd30   = c1.number_input("DPD-30 (12m)", value=int(b.get("dpd_30_count", 0)), min_value=0)
        dpd90   = c2.number_input("DPD-90 (12m)", value=int(b.get("dpd_90_count", 0)), min_value=0)
        enq     = c3.number_input("Enquiries (6m)", value=int(b.get("enquiries_last_6m", 1)), min_value=0)
        existing_emi = st.number_input("Existing EMI (₹/month)", value=int(b.get("total_existing_emi", 0)), min_value=0, step=500)
        wo      = st.checkbox("Written-Off Account", value=bool(b.get("written_off_flag", False)))

        submitted = st.form_submit_button("Run Underwriting →", use_container_width=True, type="primary")


# ── RIGHT: Live agent trace ───────────────────────────────────────────────────

with col_trace:
    st.subheader("Live Agent Trace")
    trace_container = st.empty()
    decision_container = st.empty()


# ── Run underwriting ──────────────────────────────────────────────────────────

if submitted:
    profile_dict = {
        "name": name, "age": age, "employment_type": emp,
        "monthly_income": income, "employer": "", "city": city,
        "state": state_, "pin_code": "400001",
        "loan_amount": loan, "loan_tenure_months": tenure,
        "loan_purpose": purpose, "requested_emi": emi,
    }
    bureau_dict = {
        "cibil_score": cibil, "credit_utilization": util,
        "dpd_30_count": int(dpd30), "dpd_90_count": int(dpd90),
        "enquiries_last_6m": int(enq), "total_existing_emi": existing_emi,
        "oldest_account_months": 36, "written_off_flag": wo,
        "total_accounts": 3, "active_accounts": 2, "secured_unsecured_ratio": 0.5,
    }
    stmts = _gen_statements(profile_dict)
    state = initial_state(
        application_id    = f"UI-{int(time.time())}",
        applicant_profile = profile_dict,
        bureau_report     = bureau_dict,
        bank_statements   = stmts,
    )

    agent_order = [
        "timing_start", "bureau_agent", "income_agent",
        "bank_agent", "fraud_agent", "policy_agent",
        "decision_agent", "report_agent", "timing_end",
    ]
    agent_icons = {
        "bureau_agent": "📊 Bureau Agent",
        "income_agent": "💰 Income Agent",
        "bank_agent":   "🏧 Bank Statement Agent",
        "fraud_agent":  "🔍 Fraud Agent",
        "policy_agent": "📋 Policy RAG Agent",
        "decision_agent": "⚖️ Decision Agent",
        "report_agent": "📄 Report Agent",
    }
    completed: dict[str, str] = {}
    running_agent = ""

    def render_trace(completed: dict, running: str) -> str:
        lines = []
        for key, label in agent_icons.items():
            if key in completed:
                lines.append(f"✅ **{label}** — {completed[key]}")
            elif key == running:
                lines.append(f"⏳ **{label}** — running …")
            else:
                lines.append(f"⬜ {label}")
        return "\n\n".join(lines)

    trace_container.markdown(render_trace(completed, "bureau_agent"))

    with st.spinner("Underwriting in progress …"):
        try:
            for event in workflow.stream(state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name in agent_icons:
                        trace = node_output.get("agent_trace", [""])
                        summary = trace[0] if trace else "done"
                        completed[node_name] = summary.split("]", 1)[-1].strip()
                        next_agents = [a for a in agent_icons if a not in completed]
                        running_agent = next_agents[0] if next_agents else ""
                        trace_container.markdown(render_trace(completed, running_agent))

            result = workflow.invoke(state)
            st.session_state["result"] = result
            trace_container.markdown(render_trace(completed, ""))

        except Exception as exc:
            st.error(f"Underwriting error: {exc}")


# ── Decision display ──────────────────────────────────────────────────────────

result = st.session_state.get("result")
if result:
    st.divider()

    rec   = result.get("recommendation", "")
    score = float(result.get("credit_score", 0))
    conf  = float(result.get("confidence", 0))
    ms    = result.get("processing_time_ms", 0)

    badge_map = {
        "AUTO_APPROVE":              ("✅ AUTO APPROVE",             "#1a7a1a"),
        "APPROVE_WITH_CONDITIONS":   ("🟡 APPROVE WITH CONDITIONS",  "#8a7a00"),
        "REFER_TO_CREDIT_COMMITTEE": ("🔵 REFER TO CREDIT COMMITTEE","#1a4a8a"),
        "AUTO_REJECT":               ("❌ AUTO REJECT",              "#8a1a1a"),
    }
    label, color = badge_map.get(rec, ("❓ UNKNOWN", "#555"))

    # Decision banner
    st.markdown(
        f"<div style='background:{color};padding:16px;border-radius:8px;color:white;"
        f"font-size:22px;font-weight:bold;text-align:center'>{label}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Credit Score", f"{score:.0f} / 1000")
    m2.metric("Confidence", f"{conf:.0%}")
    m3.metric("Processing Time", f"{ms / 1000:.1f}s")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📄 Credit Memo", "📊 Score Breakdown", "🔍 Full Trace"])

    with tab1:
        st.markdown(result.get("credit_memo", "No memo generated."))

    with tab2:
        bureau_a  = result.get("bureau_analysis", {})
        income_a  = result.get("income_analysis", {})
        bank_a    = result.get("bank_statement_analysis", {})
        fraud_a   = result.get("fraud_signals", {})
        policy_a  = result.get("policy_check", {})

        b_score = float(bureau_a.get("bureau_score", 0))
        i_score = float(income_a.get("income_score", 0))
        c_score = float(bank_a.get("cashflow_score", 0))
        f_score = float(fraud_a.get("fraud_score", 0))
        p_score = float(policy_a.get("policy_score", 0))

        # Radar chart
        categories = ["Bureau", "Income", "Cashflow", "Fraud (inv.)", "Policy"]
        values     = [b_score, i_score, c_score, 1000 - f_score, p_score]

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Agent Scores",
            line_color="#3b82f6",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1000])),
            showlegend=False,
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score table
        st.markdown("| Component | Score | Weight | Contribution |")
        st.markdown("|---|---|---|---|")
        rows = [
            ("Bureau Analysis",   b_score,       0.35),
            ("Income",            i_score,       0.25),
            ("Cashflow",          c_score,       0.20),
            ("Fraud (inverted)",  1000 - f_score, 0.15),
            ("Policy",            p_score,       0.05),
        ]
        for name, s, w in rows:
            st.markdown(f"| {name} | {s:.0f}/1000 | {w:.0%} | **{s*w:.1f}** |")
        st.markdown(f"| **COMPOSITE** | | | **{score:.0f}/1000** |")

        # Flags
        all_flags = []
        for src, data in [("Bureau", bureau_a), ("Income", income_a),
                          ("Cashflow", bank_a), ("Fraud", fraud_a), ("Policy", policy_a)]:
            for f in data.get("flags", []):
                all_flags.append(f"[{src}] {f}")
        if all_flags:
            st.subheader("Flags Raised")
            for fl in all_flags:
                st.warning(fl)
        else:
            st.success("No flags raised during underwriting.")

    with tab3:
        for line in result.get("agent_trace", []):
            st.code(line, language=None)

        if result.get("errors"):
            st.error("Errors: " + "; ".join(result["errors"]))
