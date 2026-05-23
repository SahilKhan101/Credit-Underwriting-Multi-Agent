"""
LangGraph StateGraph — wires all agents into a parallel fan-out / fan-in topology.

Flow:
  START
    └─► timing_start
          ├─► bureau_agent   ─┐
          ├─► income_agent   ─┤
          ├─► bank_agent     ─┼─► decision_agent ─► report_agent ─► timing_end ─► END
          ├─► fraud_agent    ─┤
          └─► policy_agent   ─┘

bureau / income / bank / fraud / policy run in parallel (LangGraph fan-out).
decision_agent waits for all five (fan-in) before executing.
"""
from __future__ import annotations

import time

from langgraph.graph import StateGraph, START, END

from graph.state import UnderwritingState
from agents.bureau_agent import bureau_analysis_node
from agents.income_agent import income_verification_node
from agents.bank_statement_agent import bank_statement_node
from agents.fraud_agent import fraud_detection_node
from agents.policy_rag_agent import policy_rag_node
from agents.decision_agent import decision_agent_node
from agents.report_agent import report_generation_node


def _timing_start(state: UnderwritingState) -> dict:
    return {"processing_start_ms": int(time.time() * 1000)}


def _timing_end(state: UnderwritingState) -> dict:
    start = state.get("processing_start_ms", int(time.time() * 1000))
    elapsed = int(time.time() * 1000) - start
    return {"processing_time_ms": elapsed}


def build_graph() -> StateGraph:
    g = StateGraph(UnderwritingState)

    # Timing bookends
    g.add_node("timing_start",  _timing_start)
    g.add_node("timing_end",    _timing_end)

    # Analysis agents
    g.add_node("bureau_agent",  bureau_analysis_node)
    g.add_node("income_agent",  income_verification_node)
    g.add_node("bank_agent",    bank_statement_node)
    g.add_node("fraud_agent",   fraud_detection_node)
    g.add_node("policy_agent",  policy_rag_node)

    # Decision + report
    g.add_node("decision_agent", decision_agent_node)
    g.add_node("report_agent",   report_generation_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    g.add_edge(START, "timing_start")

    # Fan-out: timing_start → 5 parallel analysis agents
    for agent in ("bureau_agent", "income_agent", "bank_agent", "fraud_agent", "policy_agent"):
        g.add_edge("timing_start", agent)

    # Fan-in: all 5 → decision_agent (LangGraph waits for all predecessors)
    for agent in ("bureau_agent", "income_agent", "bank_agent", "fraud_agent", "policy_agent"):
        g.add_edge(agent, "decision_agent")

    g.add_edge("decision_agent", "report_agent")
    g.add_edge("report_agent",   "timing_end")
    g.add_edge("timing_end",     END)

    return g


# Compiled graph — import this in API, UI, eval runner, main.py
app = build_graph().compile()
