"""
Generates synthetic loan applications for development, testing, and the evaluation suite.

Usage:
    python scripts/generate_synthetic_data.py --count 500 --out data/synthetic
    python scripts/generate_synthetic_data.py --count 200 --out data/eval_set --labeled
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

fake = Faker("en_IN")
rng  = np.random.default_rng(seed=42)

CITIES_TIER1 = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
CITIES_TIER2 = ["Nagpur", "Surat", "Jaipur", "Lucknow", "Chandigarh", "Indore", "Bhopal", "Kochi"]
EMPLOYMENT_TYPES = ["SALARIED", "SELF_EMPLOYED", "BUSINESS", "PROFESSIONAL", "GOVERNMENT"]
LOAN_PURPOSES = ["HOME_RENOVATION", "EDUCATION", "MEDICAL", "VEHICLE", "PERSONAL", "BUSINESS"]

STATE_PIN_MAP = {
    "Maharashtra": "41", "Karnataka": "56", "Delhi": "11",
    "Telangana": "50", "Tamil Nadu": "60", "Gujarat": "38",
    "Rajasthan": "30", "Uttar Pradesh": "20", "West Bengal": "70",
}


# ── Applicant profile ──────────────────────────────────────────────────────────

def gen_profile(segment: str = "prime") -> dict:
    """segment: 'prime' | 'near_prime' | 'subprime'"""
    emp  = random.choice(EMPLOYMENT_TYPES)
    tier = random.choice(["tier1", "tier2"])
    city = random.choice(CITIES_TIER1 if tier == "tier1" else CITIES_TIER2)
    state = {
        "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
        "Bangalore": "Karnataka", "Delhi": "Delhi", "Hyderabad": "Telangana",
        "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Ahmedabad": "Gujarat",
        "Surat": "Gujarat", "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh",
        "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
        "Chandigarh": "Punjab", "Kochi": "Kerala",
    }.get(city, "Maharashtra")

    prefix = STATE_PIN_MAP.get(state, "40")
    bad_pin = segment == "subprime" and random.random() < 0.15
    pin = (f"{prefix}{rng.integers(100000, 999999)}"[: 6]
           if not bad_pin
           else f"99{rng.integers(1000, 9999)}")

    income_base = {
        "prime":      rng.integers(50_000, 250_000),
        "near_prime": rng.integers(25_000, 80_000),
        "subprime":   rng.integers(15_000, 35_000),
    }[segment]

    age    = int(rng.integers(24, 55))
    income = int(income_base)

    # Generate existing obligations first so new loan EMI stays within FOIR budget.
    # These ratios are echoed into gen_bureau via "_existing_emi_ratio" to stay consistent.
    existing_emi_ratio = float(rng.uniform(
        *{"prime": (0.05, 0.18), "near_prime": (0.10, 0.30), "subprime": (0.15, 0.45)}[segment]
    ))
    existing_emi = int(income * existing_emi_ratio)

    # Target total FOIR bands that produce a realistic mix of outcomes
    target_total_foir = float(rng.uniform(
        *{"prime": (0.20, 0.50), "near_prime": (0.30, 0.65), "subprime": (0.45, 0.90)}[segment]
    ))
    new_emi_foir = max(0.05, target_total_foir - existing_emi_ratio)
    requested_emi = int(income * new_emi_foir)

    # Back-calculate loan amount from EMI
    tenure_months = int(random.choice([24, 36, 48, 60, 72, 84]))
    r_monthly = 0.015   # proxy 18% p.a.
    annuity_factor = (1 - (1 + r_monthly) ** -tenure_months) / r_monthly
    loan_amount = int(requested_emi * annuity_factor)

    return {
        "name":               fake.name(),
        "age":                age,
        "employment_type":    emp,
        "monthly_income":     income,
        "employer":           fake.company() if emp in ("SALARIED", "GOVERNMENT") else fake.company() + " (Self)",
        "city":               city,
        "state":              state,
        "pin_code":           pin,
        "loan_amount":        loan_amount,
        "loan_tenure_months": tenure_months,
        "loan_purpose":       random.choice(LOAN_PURPOSES),
        "requested_emi":      requested_emi,
        "_existing_emi":      existing_emi,   # passed to gen_bureau for consistency
    }


# ── Bureau report ──────────────────────────────────────────────────────────────

def gen_bureau(profile: dict, segment: str) -> dict:
    income = profile["monthly_income"]

    cibil = {
        "prime":      int(np.clip(rng.normal(750, 30), 700, 900)),
        "near_prime": int(np.clip(rng.normal(670, 35), 600, 749)),
        "subprime":   int(np.clip(rng.normal(580, 50), 300, 649)),
    }[segment]

    utilization = float(np.clip(rng.beta(2, 5), 0.05, 0.95))
    if segment == "subprime":
        utilization = float(np.clip(utilization + 0.2, 0.1, 0.95))

    dpd_30 = {
        "prime":      int(rng.integers(0, 1)),
        "near_prime": int(rng.integers(0, 3)),
        "subprime":   int(rng.integers(0, 5)),
    }[segment]

    dpd_90 = {
        "prime":      0,
        "near_prime": int(rng.integers(0, 1)),
        "subprime":   int(rng.integers(0, 2)),
    }[segment]

    written_off = segment == "subprime" and random.random() < 0.20
    enquiries   = int(rng.integers(0, 5 if segment != "subprime" else 8))
    oldest      = int(rng.integers(6, 96))
    # Use pre-computed existing_emi from gen_profile to keep FOIR consistent
    existing_emi = profile.get("_existing_emi", int(income * rng.uniform(0.05, 0.20)))

    return {
        "cibil_score":          cibil,
        "total_accounts":       int(rng.integers(1, 8)),
        "active_accounts":      int(rng.integers(1, 5)),
        "credit_utilization":   round(utilization, 4),
        "dpd_30_count":         dpd_30,
        "dpd_90_count":         dpd_90,
        "enquiries_last_6m":    enquiries,
        "oldest_account_months": oldest,
        "total_existing_emi":   existing_emi,
        "written_off_flag":     written_off,
        "secured_unsecured_ratio": round(float(rng.uniform(0.2, 0.8)), 2),
    }


# ── Bank statements (6 months) ─────────────────────────────────────────────────

def gen_bank_statements(profile: dict, segment: str) -> list[dict]:
    income    = profile["monthly_income"]
    req_emi   = profile["requested_emi"]
    start_dt  = datetime.today() - timedelta(days=180)

    transactions = []
    balance = income * float(rng.uniform(1.5, 4.0))

    for m in range(6):
        month_start = start_dt + timedelta(days=m * 30)

        # Salary credit — sometimes irregular for near_prime/subprime
        salary_date = month_start + timedelta(days=int(rng.integers(1, 5)))
        salary_amt  = int(income * float(rng.uniform(0.95, 1.05)))
        if segment == "subprime" and random.random() < 0.20:
            # Missing salary month
            pass
        else:
            balance += salary_amt
            transactions.append({
                "date": salary_date.strftime("%Y-%m-%d"),
                "description": "SALARY CR NEFT",
                "amount": salary_amt,
                "type": "credit",
                "balance_after": round(balance, 2),
                "bounced": False,
            })

        # EMI debit
        emi_date = month_start + timedelta(days=5)
        bounced  = segment == "subprime" and random.random() < 0.12
        balance -= req_emi if not bounced else 0
        transactions.append({
            "date":  emi_date.strftime("%Y-%m-%d"),
            "description": "EMI DEBIT",
            "amount": req_emi,
            "type": "debit",
            "balance_after": round(balance, 2),
            "bounced": bounced,
        })

        # Utility bills
        for _ in range(int(rng.integers(2, 5))):
            bill_date = month_start + timedelta(days=int(rng.integers(1, 28)))
            bill_amt  = int(rng.integers(500, 8000))
            balance   -= bill_amt
            transactions.append({
                "date": bill_date.strftime("%Y-%m-%d"),
                "description": random.choice(["UTILITY BILL", "INSURANCE", "MOBILE BILL", "OTT SUBSCRIPTION"]),
                "amount": bill_amt,
                "type": "debit",
                "balance_after": round(balance, 2),
                "bounced": False,
            })

        # Cash withdrawals — higher for subprime (fraud signal)
        cash_multiplier = {"prime": 0.05, "near_prime": 0.10, "subprime": 0.35}[segment]
        cash_amt = int(income * float(rng.uniform(0, cash_multiplier * 2)))
        if cash_amt > 0:
            balance -= cash_amt
            transactions.append({
                "date": (month_start + timedelta(days=int(rng.integers(10, 20)))).strftime("%Y-%m-%d"),
                "description": "ATM CASH WITHDRAWAL",
                "amount": cash_amt,
                "type": "debit",
                "balance_after": round(balance, 2),
                "bounced": False,
            })

        balance = max(balance, 500)   # prevent deeply negative balance

    return sorted(transactions, key=lambda t: t["date"])


# ── Ground-truth label ─────────────────────────────────────────────────────────

def derive_label(_segment: str, bureau: dict, profile: dict) -> str:
    cibil  = bureau["cibil_score"]
    dpd90  = bureau["dpd_90_count"]
    wo     = bureau["written_off_flag"]
    income = profile["monthly_income"]
    req_emi = profile["requested_emi"]
    existing_emi = bureau["total_existing_emi"]
    foir   = (existing_emi + req_emi) / max(income, 1)

    if dpd90 > 0 or wo or cibil < 550:
        return "AUTO_REJECT"
    if cibil >= 750 and foir <= 0.45:
        return "AUTO_APPROVE"
    if cibil >= 650 and foir <= 0.50:
        return "APPROVE_WITH_CONDITIONS"
    if cibil >= 550 and foir <= 0.60:
        return "REFER_TO_CREDIT_COMMITTEE"
    return "AUTO_REJECT"


# ── Main generator ─────────────────────────────────────────────────────────────

def generate(count: int, out_dir: str, labeled: bool = False) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    applications = []

    # Distribution: 60% prime, 25% near_prime, 15% subprime (eval: 50/25/25)
    if labeled:
        segments = (
            ["prime"] * (count // 2)
            + ["subprime"] * (count // 4)
            + ["near_prime"] * (count - count // 2 - count // 4)
        )
    else:
        segments = (
            ["prime"] * int(count * 0.60)
            + ["near_prime"] * int(count * 0.25)
            + ["subprime"] * (count - int(count * 0.60) - int(count * 0.25))
        )
    random.shuffle(segments)

    for i, segment in enumerate(segments):
        app_id  = f"APP_{datetime.today().strftime('%Y%m')}_{i+1:04d}"
        profile = gen_profile(segment)
        bureau  = gen_bureau(profile, segment)
        stmts   = gen_bank_statements(profile, segment)
        label   = derive_label(segment, bureau, profile) if labeled else None

        # Strip internal coordination key before persisting
        profile.pop("_existing_emi", None)

        record = {
            "application_id":  app_id,
            "segment":         segment,
            "applicant_profile": profile,
            "bureau_report":   bureau,
            "bank_statements": stmts,
        }
        if label:
            record["expected_decision"] = label

        applications.append(record)
        # Save individual JSON
        path = Path(out_dir) / f"{app_id}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    print(f"Generated {count} applications → {out_dir}/")
    if labeled:
        from collections import Counter
        labels = Counter(a["expected_decision"] for a in applications)
        print("Label distribution:", dict(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",   type=int, default=500, help="Number of applications to generate")
    parser.add_argument("--out",     type=str, default="data/synthetic")
    parser.add_argument("--labeled", action="store_true", help="Add ground-truth expected_decision labels")
    args = parser.parse_args()
    generate(args.count, args.out, args.labeled)
