"""
CLI entry point — run a single loan application through the underwriting system.

Usage:
    python main.py                              # use built-in demo application
    python main.py --file data/synthetic/APP_202501_0001.json
    python main.py --segment prime              # generate and run a prime applicant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph.state import initial_state
from graph.workflow import app as workflow


def run_demo(segment: str = "prime") -> None:
    from scripts.generate_synthetic_data import gen_profile, gen_bureau, gen_bank_statements
    profile = gen_profile(segment)
    bureau  = gen_bureau(profile, segment)
    stmts   = gen_bank_statements(profile, segment)
    run_application("DEMO-001", profile, bureau, stmts)


def run_file(path: str) -> None:
    with open(path) as f:
        record = json.load(f)
    run_application(
        application_id    = record["application_id"],
        applicant_profile = record["applicant_profile"],
        bureau_report     = record["bureau_report"],
        bank_statements   = record.get("bank_statements", []),
    )


def run_application(
    application_id: str,
    applicant_profile: dict,
    bureau_report: dict,
    bank_statements: list,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  CreditIQ — Underwriting Application {application_id}")
    print(f"{'─'*60}")
    print(f"  Applicant : {applicant_profile.get('name')} | Age {applicant_profile.get('age')}")
    print(f"  Income    : ₹{applicant_profile.get('monthly_income', 0):,.0f}/month")
    print(f"  Loan Ask  : ₹{applicant_profile.get('loan_amount', 0):,.0f} × {applicant_profile.get('loan_tenure_months')}m")
    print(f"  CIBIL     : {bureau_report.get('cibil_score')}")
    print(f"\n  Running agents …\n")

    state = initial_state(
        application_id    = application_id,
        applicant_profile = applicant_profile,
        bureau_report     = bureau_report,
        bank_statements   = bank_statements,
    )

    for event in workflow.stream(state, stream_mode="updates"):
        for node, output in event.items():
            for line in output.get("agent_trace", []):
                print(f"  {line}")

    result = workflow.invoke(state)

    print(f"\n{'='*60}")
    print(f"  DECISION : {result.get('recommendation')}")
    print(f"  Score    : {result.get('credit_score', 0):.0f} / 1000")
    print(f"  Confidence: {result.get('confidence', 0):.0%}")
    print(f"  Latency  : {result.get('processing_time_ms', 0)}ms")
    print(f"{'='*60}")
    print(f"\n{result.get('decision_rationale', '')}\n")

    if result.get("key_strengths"):
        print("Strengths:")
        for s in result["key_strengths"]:
            print(f"  + {s}")

    if result.get("key_risks"):
        print("\nRisks:")
        for r in result["key_risks"]:
            print(f"  ! {r}")

    if result.get("conditions"):
        print("\nConditions for approval:")
        for c in result["conditions"]:
            print(f"  → {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditIQ CLI")
    parser.add_argument("--file",    type=str, help="Path to application JSON file")
    parser.add_argument("--segment", type=str, default="prime",
                        choices=["prime", "near_prime", "subprime"],
                        help="Synthetic applicant segment to generate")
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
    else:
        run_demo(args.segment)
