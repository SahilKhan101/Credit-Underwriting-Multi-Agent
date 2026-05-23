"""
Batch evaluation runner — processes the labeled eval set (200 cases) and
reports all model-level and system-level metrics.

Usage:
    # First generate the labeled eval set:
    python scripts/generate_synthetic_data.py --count 200 --out data/eval_set --labeled

    # Then run evaluation:
    python evaluation/eval_runner.py --eval-dir data/eval_set --sample 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root to sys.path when run directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import initial_state
from graph.workflow import app as workflow
from evaluation.metrics import summary_report


# Map expected_decision labels to binary bad flag (1=bad/rejected, 0=good)
_BAD_MAP = {
    "AUTO_REJECT":               1,
    "REFER_TO_CREDIT_COMMITTEE": 1,   # treat refers as bad for Gini/KS
    "APPROVE_WITH_CONDITIONS":   0,
    "AUTO_APPROVE":              0,
}


def run_single(record: dict) -> dict[str, Any]:
    state = initial_state(
        application_id   = record["application_id"],
        applicant_profile = record["applicant_profile"],
        bureau_report    = record["bureau_report"],
        bank_statements  = record.get("bank_statements", []),
    )
    t0     = time.time()
    result = workflow.invoke(state)
    elapsed = round((time.time() - t0) * 1000)

    return {
        "application_id":  record["application_id"],
        "expected":        record.get("expected_decision"),
        "predicted":       result.get("recommendation"),
        "credit_score":    result.get("credit_score", 0),
        "confidence":      result.get("confidence", 0),
        "foir":            result.get("bureau_analysis", {}).get("foir", 0),
        "latency_ms":      elapsed,
        "errors":          result.get("errors", []),
    }


def run_eval(eval_dir: str, sample: int | None = None) -> None:
    eval_path = Path(eval_dir)
    files     = sorted(eval_path.glob("*.json"))

    if sample:
        import random
        random.shuffle(files)
        files = files[:sample]

    print(f"\nRunning evaluation on {len(files)} applications from {eval_dir} …\n")

    results: list[dict] = []
    errors  = 0

    for fp in tqdm(files, unit="app"):
        with open(fp) as f:
            record = json.load(f)
        try:
            r = run_single(record)
            results.append(r)
            if r["errors"]:
                errors += 1
        except Exception as exc:
            print(f"  ERROR on {fp.name}: {exc}")
            errors += 1

    if not results:
        print("No results collected.")
        return

    # ── Compile metric inputs ──────────────────────────────────────────────
    scores       = [r["credit_score"] for r in results]
    recommendations = [r["predicted"] for r in results]
    expected     = [r["expected"] for r in results if r["expected"]]
    foiars       = [r["foir"] for r in results]
    y_true       = [_BAD_MAP.get(r["expected"] or "REFER_TO_CREDIT_COMMITTEE", 1) for r in results]

    metrics = summary_report(scores, recommendations, y_true, foiars, expected or None)

    latencies = [r["latency_ms"] for r in results]
    metrics["avg_latency_ms"]    = round(sum(latencies) / len(latencies))
    metrics["p95_latency_ms"]    = round(sorted(latencies)[int(len(latencies) * 0.95)])
    metrics["error_rate"]        = round(errors / len(files), 4)

    # ── Print report ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CREDIT UNDERWRITING SYSTEM — EVALUATION REPORT")
    print("=" * 55)

    field_map = {
        "n":                     "Applications evaluated",
        "approval_rate":         "Approval rate (≥650 score)",
        "auto_decision_rate":    "Auto-decision rate",
        "gini_coefficient":      "Gini coefficient (target >0.45)",
        "ks_statistic":          "KS statistic     (target >0.35)",
        "foir_compliance_rate":  "FOIR compliance (approved cases)",
        "decision_agreement_rate": "Decision agreement vs. labels",
        "avg_latency_ms":        "Avg latency (ms)",
        "p95_latency_ms":        "P95 latency (ms)",
        "error_rate":            "Error rate",
    }
    for key, label in field_map.items():
        val = metrics.get(key)
        if val is not None:
            print(f"  {label:<42}  {val}")

    print("\n  Bad rate by score band (lower band → higher bad rate = well-calibrated):")
    for band, rate in metrics.get("bad_rate_by_band", {}).items():
        import math
        if rate is None or (isinstance(rate, float) and math.isnan(rate)):
            print(f"    [{band:>9}]  n/a  (no cases in band)")
        else:
            bar = "█" * int(rate * 20)
            print(f"    [{band:>9}]  {rate:.2%}  {bar}")

    print("\n  Recommendation distribution:")
    from collections import Counter
    for rec, cnt in Counter(recommendations).most_common():
        pct = cnt / len(recommendations)
        print(f"    {rec:<32}  {cnt:>4}  ({pct:.0%})")

    print("=" * 55)

    # Save results JSON
    out_path = eval_path / "_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval_set")
    parser.add_argument("--sample",   type=int, default=None, help="Run on a random subset")
    args = parser.parse_args()
    run_eval(args.eval_dir, args.sample)
