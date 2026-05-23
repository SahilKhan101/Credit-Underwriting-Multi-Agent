# CreditIQ — Comprehensive Technical Report

> A deep-dive into every component of the multi-agent credit underwriting system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [LangGraph: State, Nodes, and Edges](#3-langgraph-state-nodes-and-edges)
4. [Data: How It Is Generated and Why](#4-data-how-it-is-generated-and-why)
5. [Agent Deep-Dives](#5-agent-deep-dives)
   - 5.1 Bureau Analysis Agent
   - 5.2 Income Verification Agent
   - 5.3 Bank Statement Agent
   - 5.4 Fraud Detection Agent
   - 5.5 Policy RAG Agent
   - 5.6 Decision Agent
   - 5.7 Report Agent
6. [RAG Pipeline: Chunking, Embeddings, Retrieval](#6-rag-pipeline-chunking-embeddings-retrieval)
7. [Composite Scoring and Decision Logic](#7-composite-scoring-and-decision-logic)
8. [LLM Layer](#8-llm-layer)
9. [Entry Points: CLI, API, UI](#9-entry-points-cli-api-ui)
10. [Evaluation Suite](#10-evaluation-suite)
11. [Policy Documents](#11-policy-documents)
12. [Configuration and Constants](#12-configuration-and-constants)
13. [Test Suite](#13-test-suite)
14. [What Can Be Improved](#14-what-can-be-improved)

---

## 1. Project Overview

**CreditIQ** is a portfolio-grade multi-agent system that autonomously underwrites Indian personal and MSME loan applications. It is modelled on a real NBFC (Non-Banking Financial Company) credit workflow.

**Domain context:**
- Indian retail credit uses CIBIL scores (300–900 scale, equivalent to FICO in the US).
- Regulatory compliance is governed by RBI (Reserve Bank of India) fair practice guidelines.
- Key risk metrics used in India: FOIR (Fixed Obligation to Income Ratio), DPD (Days Past Due), LTI (Loan-to-Income ratio).

**Technology core:**
- **LangGraph** — orchestrates the agents as a stateful directed graph with parallel fan-out.
- **LangChain** — prompt templates, chain composition, structured output.
- **Google AI Studio (Gemini)** — LLM provider for narrative generation and structured decisions.
- **ChromaDB + HuggingFace Embeddings** — local RAG vectorstore for policy documents.
- **FastAPI** — REST interface.
- **Streamlit** — interactive demo UI with live agent trace.

---

## 2. System Architecture

### 2.1 Graph Topology

```
START
  └─► timing_start
            ├─► bureau_agent   ─┐
            ├─► income_agent   ─┤
            ├─► bank_agent     ─┼─► decision_agent ─► report_agent ─► timing_end ─► END
            ├─► fraud_agent    ─┤
            └─► policy_agent   ─┘
               (all 5 parallel)
```

Five analysis agents run simultaneously (LangGraph fan-out). The decision agent waits for all five to complete (fan-in). This means the wall-clock latency for the analysis phase is bounded by the slowest single agent, not the sum.

### 2.2 Component Map

```
creditUnderwritingAgent/
├── agents/                 ← One file per agent, each self-contained
├── graph/
│   ├── state.py            ← UnderwritingState TypedDict (shared memory)
│   └── workflow.py         ← StateGraph wiring (edges)
├── data/
│   ├── policy_docs/        ← 2 Markdown policy files (RAG source)
│   ├── synthetic/          ← Generated at runtime
│   └── eval_set/           ← Labeled generated cases
├── vectorstore/            ← ChromaDB on disk (built at first run)
├── models/                 ← Placeholder for XGBoost (not yet trained)
├── api/main.py             ← FastAPI endpoints
├── ui/app.py               ← Streamlit demo
├── evaluation/             ← Batch eval + metrics
├── scripts/                ← Data generation + vectorstore warm-up
├── utils/llm.py            ← LLM factory with rate limiter
├── config.py               ← All constants and thresholds
└── main.py                 ← CLI entry point
```

### 2.3 Data Flow

Every input — whether from the synthetic generator, Streamlit form, REST API, or a saved JSON file — is normalized into the same `UnderwritingState` dict before the graph runs. Agents never know the data's origin.

```
┌─── SOURCE ──────────────────────────────────────────────────┐
│  python main.py --segment prime                             │
│  streamlit run ui/app.py  (form submission)                 │
│  POST /underwrite  (REST body)                              │
│  eval_runner.py --eval-dir data/eval_set  (batch)           │
└─────────────────────────────────────────────────────────────┘
                          │
               initial_state() factory
                          │
          UnderwritingState dict (all agent output fields absent)
                          │
                    LangGraph graph
                          │
              ┌───────────┴────────────┐
         stream events             invoke()
         (UI live trace)         (API/CLI/eval)
```

---

## 3. LangGraph: State, Nodes, and Edges

### 3.1 UnderwritingState (`graph/state.py`)

`UnderwritingState` is a `TypedDict` with `total=False` — all fields are optional at the type level. This is intentional: when the graph starts, only the four input fields exist. Agent output fields get populated as nodes execute.

```python
class UnderwritingState(TypedDict, total=False):
    # ── Required inputs ────────────────────────────────────────
    application_id: str
    applicant_profile: dict        # name, age, employment, income, loan ask, city
    bureau_report: dict            # CIBIL score, DPD, utilization, EMI, enquiries
    bank_statements: List[dict]    # list of 6-month transactions

    # ── Agent outputs (filled progressively) ──────────────────
    bureau_analysis: dict
    income_analysis: dict
    bank_statement_analysis: dict
    fraud_signals: dict
    policy_check: dict

    # ── Decision layer ─────────────────────────────────────────
    credit_score: float            # composite 0–1000
    recommendation: str
    confidence: float
    decision_rationale: str
    key_strengths: List[str]
    key_risks: List[str]
    conditions: List[str]
    credit_memo: str

    # ── Metadata ───────────────────────────────────────────────
    agent_trace: Annotated[List[str], operator.add]   # reducer: append-merge
    errors: Annotated[List[str], operator.add]        # reducer: append-merge
    processing_start_ms: int
    processing_time_ms: int
```

**Key design detail — reducers:** `agent_trace` and `errors` use `Annotated[List[str], operator.add]`. This is a LangGraph reducer annotation. When five parallel agents each return an `agent_trace` list, LangGraph uses `operator.add` (list concatenation) to merge them without conflict, instead of the default behavior of last-writer-wins. This is how parallel nodes can safely write to the same state field.

### 3.2 initial_state() factory

```python
def initial_state(application_id, applicant_profile, bureau_report, bank_statements):
    return UnderwritingState(
        application_id=application_id,
        applicant_profile=applicant_profile,
        bureau_report=bureau_report,
        bank_statements=bank_statements,
        agent_trace=[],
        errors=[],
    )
```

Explicitly initializes the reducer fields to empty lists. Without this, the first parallel append would fail.

### 3.3 Workflow Graph (`graph/workflow.py`)

```python
g = StateGraph(UnderwritingState)

# Timing bookends — track wall-clock latency
g.add_node("timing_start", _timing_start)
g.add_node("timing_end",   _timing_end)

# Analysis agents
g.add_node("bureau_agent",  bureau_analysis_node)
g.add_node("income_agent",  income_verification_node)
g.add_node("bank_agent",    bank_statement_node)
g.add_node("fraud_agent",   fraud_detection_node)
g.add_node("policy_agent",  policy_rag_node)

# Decision + report
g.add_node("decision_agent", decision_agent_node)
g.add_node("report_agent",   report_generation_node)

# Fan-out: START → timing_start → 5 agents (all in parallel)
g.add_edge(START, "timing_start")
for agent in ("bureau_agent", "income_agent", "bank_agent", "fraud_agent", "policy_agent"):
    g.add_edge("timing_start", agent)

# Fan-in: 5 agents → decision_agent (LangGraph waits for all predecessors)
for agent in ("bureau_agent", "income_agent", "bank_agent", "fraud_agent", "policy_agent"):
    g.add_edge(agent, "decision_agent")

# Linear tail: decision → report → timing_end → END
g.add_edge("decision_agent", "report_agent")
g.add_edge("report_agent",   "timing_end")
g.add_edge("timing_end",     END)

app = g.compile()
```

**Fan-out:** When `timing_start` has multiple outgoing edges to `bureau_agent`, `income_agent`, etc., LangGraph schedules all of them to run concurrently (via async or thread pool depending on runtime).

**Fan-in:** When `decision_agent` has five incoming edges, LangGraph automatically waits for all five predecessor nodes to finish before invoking `decision_agent_node`. This is native LangGraph behavior — no explicit barrier/gate is needed.

### 3.4 Node Contract

Every agent node function has the same signature and return shape:

```python
def some_agent_node(state: UnderwritingState) -> dict:
    # Read from state
    # Compute
    # (optionally) call LLM
    return {
        "some_analysis": {...},              # new state key
        "agent_trace": ["[Agent] summary"], # reducer key: list append
    }
```

The return dict is a **partial state update**. LangGraph merges it into the full state using the reducer rules (concatenate for lists, overwrite for scalars).

---

## 4. Data: How It Is Generated and Why

### 4.1 Why Synthetic Data

No real borrower data is available for a portfolio project. The synthetic generator (`scripts/generate_synthetic_data.py`) is designed to produce statistically realistic distributions for three risk segments:

| Segment    | CIBIL Range    | Monthly Income   | DPD-90 Likelihood | Written-Off % |
|------------|----------------|------------------|--------------------|---------------|
| prime      | ~750 ± 30      | ₹50k–₹2.5L       | 0%                 | 0%            |
| near_prime | ~670 ± 35      | ₹25k–₹80k        | ~10%               | 0%            |
| subprime   | ~580 ± 50      | ₹15k–₹35k        | ~20%               | ~20%          |

### 4.2 FOIR-Aware Generation (Critical Design Detail)

The most important design decision in the generator is that **loan size and existing EMI are generated jointly, not independently**.

Early versions generated `loan_amount` from income alone and `existing_emi` from a separate random draw. This caused the total FOIR (existing + new) to exceed the 50% cap for almost every case, resulting in 162/200 eval cases being AUTO_REJECT — no discrimination signal.

The fix is a coordinated generation sequence:

```python
# Step 1: pick an existing obligation ratio for this segment
existing_emi_ratio = rng.uniform(*{
    "prime": (0.05, 0.18), "near_prime": (0.10, 0.30), "subprime": (0.15, 0.45)
}[segment])
existing_emi = int(income * existing_emi_ratio)

# Step 2: pick a target total FOIR that produces a realistic outcome mix
target_total_foir = rng.uniform(*{
    "prime": (0.20, 0.50), "near_prime": (0.30, 0.65), "subprime": (0.45, 0.90)
}[segment])

# Step 3: back-calculate what the new loan EMI should be
new_emi_foir = max(0.05, target_total_foir - existing_emi_ratio)
requested_emi = int(income * new_emi_foir)

# Step 4: back-calculate loan amount from EMI using annuity formula
# (18% p.a. monthly rate = 1.5%)
r_monthly = 0.015
annuity_factor = (1 - (1 + r_monthly) ** -tenure_months) / r_monthly
loan_amount = int(requested_emi * annuity_factor)
```

The key `_existing_emi` is passed from `gen_profile` to `gen_bureau` as an internal coordination key, then stripped before saving to JSON:

```python
# gen_bureau reads it from profile
existing_emi = profile.get("_existing_emi", int(income * rng.uniform(0.05, 0.20)))

# main generator strips it before persisting
profile.pop("_existing_emi", None)
```

This produces a realistic outcome distribution when running the labeled eval set:
- ~20% AUTO_APPROVE (prime with low FOIR)
- ~38% APPROVE_WITH_CONDITIONS (prime/near_prime with moderate FOIR)
- ~16% REFER_TO_CREDIT_COMMITTEE (borderline)
- ~26% AUTO_REJECT (subprime, DPD-90, written-off)

### 4.3 Bank Statement Generation

Six months of transaction records are generated per application. Each month contains:

- **Salary credit** — NEFT credit around the 1st–5th. Amount ±5% of stated income. Subprime applicants have a 20% chance of a missing salary month.
- **EMI debit** — on the 5th. Subprime applicants have a 12% bounce probability.
- **2–5 utility/bill debits** — random amounts ₹500–₹8,000.
- **Cash withdrawal** — ATM debit; cash multiplier is `0.05×` for prime, `0.35×` for subprime. This directly feeds the fraud detection cash withdrawal ratio check.

The `balance_after` field tracks a running balance, initialized at 1.5–4× monthly income. A floor of ₹500 prevents deeply negative balances.

### 4.4 Ground-Truth Labels (`--labeled` flag)

When generating the eval set, `derive_label()` applies a deterministic rule tree to produce the expected decision:

```python
def derive_label(_segment, bureau, profile):
    cibil  = bureau["cibil_score"]
    foir   = (bureau["total_existing_emi"] + profile["requested_emi"]) / profile["monthly_income"]

    if dpd90 > 0 or written_off or cibil < 550:
        return "AUTO_REJECT"
    if cibil >= 750 and foir <= 0.45:
        return "AUTO_APPROVE"
    if cibil >= 650 and foir <= 0.50:
        return "APPROVE_WITH_CONDITIONS"
    if cibil >= 550 and foir <= 0.60:
        return "REFER_TO_CREDIT_COMMITTEE"
    return "AUTO_REJECT"
```

These labels are used by the eval runner to compute decision agreement rate (how often the system's LLM decision matches the rule-based ground truth).

### 4.5 File Format

Each application is saved as an individual JSON file named `APP_YYYYMM_NNNN.json`:

```json
{
  "application_id": "APP_202601_0001",
  "segment": "prime",
  "applicant_profile": { "name": "...", "age": 34, "monthly_income": 95000, ... },
  "bureau_report": { "cibil_score": 762, "dpd_30_count": 0, ... },
  "bank_statements": [{ "date": "2026-01-05", "description": "SALARY CR NEFT", ... }],
  "expected_decision": "AUTO_APPROVE"   // only in labeled eval set
}
```

---

## 5. Agent Deep-Dives

Every agent follows the same structure:
1. Extract relevant fields from `state`
2. Compute rule-based scores and flags
3. Call LLM for a 2–3 sentence narrative
4. Return partial state update

### 5.1 Bureau Analysis Agent (`agents/bureau_agent.py`)

**Purpose:** Parse the CIBIL-like bureau report. Detect derogatory marks. Score creditworthiness from bureau data.

**Inputs read from state:**
- `state["bureau_report"]` — CIBIL score, DPD counts, utilization, enquiries, existing EMI, written-off flag
- `state["applicant_profile"]` — monthly income, requested EMI (to calculate FOIR)

**FOIR calculation:**
```python
foir = (existing_emi + requested_emi) / max(monthly_income, 1)
```
FOIR = (all existing obligations + new proposed EMI) ÷ gross monthly income. The 50% limit is from RBI guidelines.

**Derogatory check — hard cap:**
```python
if dpd_90 > 0 or written_off:
    derogatory = True
    flags.append("DEROGATORY: DPD-90+ or written-off account detected")
```
If derogatory, the bureau score is hard-capped at 350/1000, regardless of the CIBIL score. This prevents any chance of approval when there is a serious credit history issue.

**Scoring formula (0–1000):**

| Component | Max Points | Logic |
|---|---|---|
| CIBIL base | 400 | `(cibil - 300) / 600 × 400` — linear mapping of 300–900 CIBIL onto 0–400 |
| Credit age bonus | 50 | `min(50, oldest_account_months / 2)` |
| DPD penalty | −200 max | `dpd_30 × 20 + dpd_90 × 80 + 150 if written_off` |
| Utilization penalty | −100 max | Applies only for utilization > 30% |
| Enquiry penalty | −50 max | Applies only for enquiries > 2 |
| FOIR penalty | −100 max | Applies only for FOIR > 40% |

The quality budget (400 pts) is what remains after all penalties plus the age bonus. Combined with the CIBIL base (400 pts), the theoretical maximum is 850+ (not 1000) unless the applicant has very old accounts.

**Flags raised (examples):**
- `DEROGATORY: DPD-90+ or written-off account detected`
- `HIGH UTILIZATION: 82% (>75%)`
- `EXCESSIVE ENQUIRIES: 5 in last 6 months`
- `HIGH FOIR: 58% exceeds 50% limit`
- `LOW CIBIL: 530 below minimum 550`

**LLM narrative prompt:**
```
"Bureau snapshot: CIBIL: 762 | Utilization: 28% | DPD-30 (12m): 0 | DPD-90 (12m): 0 |
Enquiries (6m): 1 | Oldest account: 60m | Written-off: False | FOIR: 32%
Bureau score: 741/1000 | Flags: None"
```
Model: `gemini-2.5-flash-lite` (fast, high quota). Output: 2–3 sentences of professional analysis.

**Output dict:**
```python
{
    "bureau_analysis": {
        "cibil_score": 762, "credit_utilization": 0.28,
        "dpd_30_count": 0, "dpd_90_count": 0, "derogatory_flag": False,
        "foir": 0.32, "oldest_account_months": 60,
        "bureau_score": 741.0, "summary": "<LLM narrative>", "flags": [],
    },
    "agent_trace": ["[Bureau Agent] score=741/1000 flags=0 (1234ms)"],
}
```

---

### 5.2 Income Verification Agent (`agents/income_agent.py`)

**Purpose:** Verify whether the applicant's stated income is credible and sufficient for the loan ask.

**Two-part income assessment:**

**Part 1 — Credibility check (estimated vs. stated):**

The agent has a rule-based income estimator as a proxy for what a trained XGBoost model would output. It applies multiplicative adjustments to the stated income:

```python
_EMPLOYMENT_MULTIPLIERS = {
    "SALARIED":      1.00,
    "SELF_EMPLOYED": 0.85,   # haircut: higher income variability
    "BUSINESS":      0.80,
    "PROFESSIONAL":  1.10,   # doctors, CAs, engineers: premium
    "GOVERNMENT":    1.05,
}
_CITY_TIER = {
    "Mumbai": 1.3, "Delhi": 1.25, "Bangalore": 1.25,
    "Chennai": 1.15, "Pune": 1.10, ...
}
age_adj = 1.0 + max(0, (age - 25) * 0.008)   # experience premium

estimated = stated × employment_multiplier × city_tier × age_adjustment
discrepancy = abs(stated - estimated) / max(estimated, 1)
```

If discrepancy > 20% (`config.INCOME_DISCREPANCY_PCT_LIMIT`), a flag is raised. This catches applicants who overstate their income relative to what someone in their employment type and city would typically earn.

**Part 2 — Salary stability from bank statements:**

```python
salary_credits = [txn["amount"] for txn in bank_statements
                  if txn["type"] == "credit" and "salary" in txn["description"].lower()]
cv  = std(salary_credits) / mean(salary_credits)   # coefficient of variation
regularity = max(0.0, 1.0 - cv * 2)               # cv=0 → 1.0; cv≥0.5 → 0.0
```

A low coefficient of variation (consistent salary amounts) results in a high regularity score. This distinguishes a stable salaried employee from someone with erratic income deposits.

**Income scoring formula:**

| Component | Max | Logic |
|---|---|---|
| Adequacy | 400 | `(annual_income / loan_amount) / MAX_LOAN_TO_INCOME_RATIO × 400` |
| Stability | 300 | `regularity_score × 300` |
| Base | 300 | Fixed — starts every applicant with 300 pts |
| Discrepancy penalty | −200 max | Applied only if discrepancy > 20% |

**Note on XGBoost stub:** The `_estimate_income()` function is designed as a plug-in point. The comment explicitly marks it: "Rule-based income estimator (plug-in point for XGBoost model)." When `models/income_xgb.pkl` is trained and loaded, it would replace this function without changing any other code.

---

### 5.3 Bank Statement Agent (`agents/bank_statement_agent.py`)

**Purpose:** Analyze 6 months of transaction history for cashflow health signals.

**Pipeline:**
1. Load transactions into a Pandas DataFrame.
2. Extract 8 features.
3. Score against thresholds.
4. Generate LLM narrative.

**Feature extraction:**

| Feature | Calculation | Meaning |
|---|---|---|
| `avg_monthly_balance` | Mean of `balance_after` column | Average Bank Balance (ABB) |
| `bounce_rate` | `len(bounced_debits) / len(all_debits)` | Fraction of failed payments |
| `salary_regularity_score` | Months with salary credit / total months | How consistently salary is credited |
| `abb_to_emi_ratio` | `avg_balance / requested_emi` | Liquidity buffer |
| `cash_withdrawal_ratio` | ATM/cash debits / total credits | Proxy for undeclared cash income or fraud |
| `net_monthly_cashflow` | `(total_credits - total_debits) / months` | Net surplus per month |

**Cashflow scoring (each component has a partial budget):**

```python
bounce_score     = max(0, (1 - bounce_rate / 0.30))   × 200   # 0% bounce → 200pts; 30%+ → 0pts
cash_score       = max(0, (1 - cash_ratio / 0.60))    × 200   # 0% cash → 200pts; 60%+ → 0pts
regularity_score = salary_regularity                  × 250   # 100% regular → 250pts
abb_score        = min(200, abb_emi_ratio × 50)               # capped at 200pts (4× EMI = max)
cf_score         = 150 if net_cf >= 0 else max(0, 150 + net_cf/1000)  # positive flow → 150pts
```

**Key flags:**
- `HIGH BOUNCE RATE: 12% of debit transactions` — suggests payment failures
- `HIGH CASH WITHDRAWAL: 45% of total credits` — fraud signal
- `IRREGULAR SALARY CREDITS: regularity 60%` — income not regularly credited
- `LOW ABB/EMI RATIO: 1.2× (target ≥1.5×)` — insufficient liquidity buffer
- `NEGATIVE NET CASHFLOW: ₹-8,500/month` — spending exceeds income

---

### 5.4 Fraud Detection Agent (`agents/fraud_agent.py`)

**Purpose:** Run four independent rule-based checks for fraud signals. Outputs a `fraud_score` where **higher = more fraudulent** (inverted in the composite formula).

**Four checks:**

**Check 1 — Velocity (application flooding):**
```python
enq = bureau["enquiries_last_6m"]
if enq > config.ENQUIRY_VELOCITY_LIMIT:   # limit = 3
    velocity_flag = True
    # Adds 250 pts to fraud_score
```
More than 3 credit enquiries in 6 months suggests the applicant is shopping for loans frantically, often a sign of financial distress or loan stacking.

**Check 2 — Cash withdrawal pattern:**
```python
cash_out = sum(ATM/cash debits)
ratio = cash_out / total_credits
if ratio > config.CASH_WITHDRAWAL_RATIO_LIMIT:   # limit = 40%
    cash_flag = True
    # Adds 200–300 pts proportional to excess
```
A high cash withdrawal ratio means income is being converted to untracked cash immediately after crediting. This is a common indicator that the account is only active to show salary credits while actual spending happens in cash.

**Check 3 — Income–EMI consistency:**
```python
if total_existing_emi / stated_income > 0.80:
    income_flag = True
    # Adds 300 pts
```
If the applicant's bureau shows existing EMIs consuming more than 80% of their declared income, the stated income is almost certainly false. A legitimate applicant would not be able to service their current obligations at that income level.

**Check 4 — Address (PIN–state consistency):**
```python
STATE_PIN_PREFIX = {"Maharashtra": ["40","41",...], "Karnataka": ["56","57",...], ...}
if pin[:2] not in expected_prefixes_for_declared_state:
    addr_flag = True
    # Adds 150 pts
```
Indian PIN codes have a deterministic state prefix. If the declared state is Maharashtra but the PIN starts with 56 (Karnataka prefix), it's a likely KYC mismatch.

**Fraud score aggregation:**
```python
score = 0
if velocity_flag: score += 250
if cash_flag:     score += 200 + min(100, excess_ratio × 500)
if income_flag:   score += 300
if addr_flag:     score += 150
fraud_score = min(1000, score)
```

**Inversion in composite score:**
The composite formula uses `(1000 - fraud_score)` as the fraud contribution. A fraud score of 0 (no fraud) contributes 1000 × 0.15 = 150 pts. A fraud score of 900 contributes only 100 × 0.15 = 15 pts.

---

### 5.5 Policy RAG Agent (`agents/policy_rag_agent.py`)

**Purpose:** Check the loan application against internal credit policy and RBI guidelines using Retrieval-Augmented Generation (RAG). See [Section 6](#6-rag-pipeline-chunking-embeddings-retrieval) for the full RAG pipeline.

**High-level flow:**
1. Build a natural-language query from the application context.
2. Retrieve the 4 most relevant policy chunks from ChromaDB.
3. Send retrieved chunks + application details to the LLM.
4. Parse the structured response for eligibility, violations, and cited policies.
5. Compute a policy score.

**Query construction:**
```python
query = (
    f"Policy eligibility for {loan_type} loan of ₹{loan_amount:,.0f} "
    f"for a {emp_type} applicant aged {age} with monthly income ₹{monthly_income:,.0f} "
    f"and FOIR {foir:.0%}. CIBIL score {cibil_score}."
)
```

This query is designed to surface clauses about eligibility limits, FOIR caps, CIBIL minimums, and loan purpose restrictions — the four most commonly relevant policy areas.

**LLM response format (strict output structure):**
```
ELIGIBLE: yes/no
VIOLATIONS: comma-separated list or 'none'
CITED_POLICIES: comma-separated policy clause names
SUMMARY: 2-3 sentence compliance summary
```

The `_parse_compliance_response()` function parses this line-by-line into a structured dict.

**Policy scoring:**
```python
policy_score = max(0.0, 1000.0 - n_violations × 200) if eligible else 200.0
```

- 0 violations → 1000/1000
- 1 violation → 800/1000
- 2 violations → 600/1000
- 3+ violations → 400/1000
- Ineligible → 200/1000 (hard floor, not 0)

**Output dict:**
```python
{
    "policy_check": {
        "eligible": True,
        "policy_violations": ["FOIR exceeds 50% limit", "Bounce rate exceeds 3%"],
        "cited_policies": ["Section 2.2 FOIR Limits", "Section 4.3 Bounce Rate"],
        "policy_score": 600.0,
        "summary": "<2-3 sentence compliance assessment>",
        "flags": ["FOIR exceeds 50% limit", "Bounce rate exceeds 3%"],
    },
    "agent_trace": ["[Policy Agent] eligible=True violations=2 (4321ms)"],
}
```

---

### 5.6 Decision Agent (`agents/decision_agent.py`)

**Purpose:** Aggregate all five analysis dicts, compute the composite credit score, derive a rule-based recommendation, then call the LLM for a structured credit decision with rationale, strengths, risks, and conditions.

**Step 1 — Composite score:**

```python
def _composite_score(bureau, income, cashflow, fraud, policy):
    fraud_contribution = 1000.0 - fraud   # invert
    return round(clamp(
        bureau    × 0.35
        + income  × 0.25
        + cashflow × 0.20
        + fraud_contribution × 0.15
        + policy  × 0.05,
        0, 1000
    ), 2)
```

**Step 2 — Rule-based recommendation:**

```python
def _score_to_recommendation(score, policy_eligible):
    if not policy_eligible: return "AUTO_REJECT"
    if score >= 750:        return "AUTO_APPROVE"
    if score >= 650:        return "APPROVE_WITH_CONDITIONS"
    if score >= 550:        return "REFER_TO_CREDIT_COMMITTEE"
    return "AUTO_REJECT"
```

**Step 3 — LLM structured decision:**

The rule-based recommendation is shown to the LLM as a hint, but the LLM can override it if the full context warrants it. The LLM receives:
- All five agent scores and their narrative summaries
- All flags raised
- The composite score and rule-based recommendation
- Applicant and loan details

It must return a `CreditDecision` Pydantic model:

```python
class CreditDecision(BaseModel):
    recommendation: Literal["AUTO_APPROVE", "APPROVE_WITH_CONDITIONS",
                             "REFER_TO_CREDIT_COMMITTEE", "AUTO_REJECT"]
    confidence: float        # 0.0–1.0
    key_strengths: List[str]
    key_risks: List[str]
    conditions: List[str]    # conditions for APPROVE_WITH_CONDITIONS
    decision_rationale: str
```

**Critical implementation note — `method="function_calling"`:**

```python
llm = get_smart_llm().with_structured_output(CreditDecision, method="function_calling")
```

The default mode for `with_structured_output` on Gemini is JSON mode. In `gemini-2.5-flash`, thinking tokens leak into JSON string fields as hundreds of `\n` characters. This truncates the JSON before `decision_rationale` is written, causing Pydantic to fail parsing and silently triggering the rule-based fallback.

`method="function_calling"` uses the function calling API instead of JSON mode. The model's thinking tokens are separated from the structured output, preventing the bleed-through.

**Fallback behavior:**

If the LLM call fails for any reason:
```python
except Exception as exc:
    rec       = rule_rec       # use the rule-based recommendation
    conf      = 0.75
    rationale = f"Rule-based decision (LLM unavailable: {exc})"
    strengths = []
    risks     = all_flags[:3]
    conditions = []
```

The system degrades gracefully — a rule-based decision is produced even without LLM access.

---

### 5.7 Report Agent (`agents/report_agent.py`)

**Purpose:** Assemble the final credit memo — the formal document handed to the credit committee or stored in the audit trail.

**Two-part output:**

**Part 1 — Deterministic template:**
The memo skeleton is assembled in pure Python: applicant details, decision banner, score breakdown table, all flags, and agent audit trail. These sections do not rely on the LLM.

**Score breakdown table:**
```python
def _score_breakdown_table(bureau, income, cashflow, fraud, policy, total):
    fraud_contrib = 1000.0 - fraud
    rows = [
        ("Bureau Analysis",              bureau,       0.35),
        ("Income Verification",          income,       0.25),
        ("Bank Statement / Cashflow",    cashflow,     0.20),
        ("Fraud Assessment (inverted)",  fraud_contrib, 0.15),
        ("Policy Compliance",            policy,       0.05),
    ]
    # → Markdown table with raw score, weight, contribution for each
```

**Part 2 — LLM narrative (4 paragraphs):**
```
Write a focused 3–4 paragraph narrative covering:
(1) applicant profile & loan request
(2) credit risk analysis
(3) income & cashflow assessment
(4) recommendation rationale
Use professional banking language. Cite specific numbers. No markdown headers.
```

The LLM receives all five agent summaries plus the decision, and synthesizes them into a cohesive memo narrative that reads as if written by a credit analyst.

**Error handling — quota awareness:**
```python
except Exception as exc:
    err_str = str(exc)
    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
        reason = "LLM daily quota exhausted — narrative unavailable. Rule-based decision is still valid."
    else:
        reason = f"LLM unavailable: {err_str[:120]}"
    narrative = f"*{reason}*\n\nDecision: **{rec}** | Score: **{score:.0f}/1000**"
```

The memo is still complete and valid even if the narrative section is unavailable — all other sections are fully deterministic.

---

## 6. RAG Pipeline: Chunking, Embeddings, Retrieval

### 6.1 Policy Documents

Two Markdown files are the RAG knowledge base:

| File | Content |
|---|---|
| `internal_credit_policy.md` | 8 sections: eligibility criteria, FOIR limits, LTI limits, bank statement requirements, derogatory marks, fraud prevention, interest rates, decision authority |
| `rbi_fair_practice_guidelines.md` | RBI-based fair practice requirements for NBFCs |

### 6.2 Chunking

Performed by `RecursiveCharacterTextSplitter`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # characters per chunk
    chunk_overlap=80,     # overlap to preserve context across chunk boundaries
    separators=["\n\n", "\n", ".", " "],   # split preference order
)
```

The separator hierarchy tries to split at paragraph breaks first, then line breaks, then sentence boundaries, then word boundaries. This preserves semantic coherence — a policy clause is unlikely to be cut in the middle of its rule text.

Each chunk is stored as a `Document` with metadata:
```python
Document(
    page_content=chunk_text,
    metadata={"source": "internal_credit_policy.md", "chunk": 5}
)
```

### 6.3 Embeddings

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

`all-MiniLM-L6-v2` is a 22M parameter sentence transformer that produces 384-dimensional dense embeddings. Key properties:
- **Local** — no API key, no network call, no per-token cost.
- **~80MB** — small enough to be practical on a laptop.
- **Normalized** — outputs unit vectors, making cosine similarity equivalent to dot product.

### 6.4 Vectorstore: ChromaDB

```python
@lru_cache(maxsize=1)
def _get_vectorstore():
    if persist_dir exists and not empty:
        return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
    # Otherwise: chunk policy docs, embed, persist
    vs = Chroma.from_documents(docs, embedding=get_embeddings(), persist_directory=persist_dir)
    return vs
```

**Lazy initialization:** The vectorstore is built on first `policy_rag_node` call and cached in memory via `@lru_cache`. If the `./vectorstore/policy_chroma/` directory already exists on disk, it is loaded instead of rebuilt. This means startup is instant after the first run.

**Retrieval:**
```python
docs = vs.similarity_search(query, k=4)
```

Returns the 4 chunks with highest cosine similarity to the query embedding. These are concatenated with `---` separators and inserted into the LLM prompt as context.

### 6.5 Why RAG Instead of Full-Context?

Both policy documents together are ~250KB of text. While a modern LLM context window could fit them, RAG has three advantages here:
1. **Precision** — the LLM sees only the 4 most relevant clauses, reducing hallucination risk.
2. **Scalability** — a real NBFC has dozens of policy documents; full-context would not scale.
3. **Auditability** — cited policy clauses are traceable to specific document sections.

---

## 7. Composite Scoring and Decision Logic

### 7.1 Score Formula

```
Credit Score (0–1000) =
    0.35 × bureau_score
  + 0.25 × income_score
  + 0.20 × cashflow_score
  + 0.15 × (1000 − fraud_score)   ← inverted
  + 0.05 × policy_score
```

Weights reflect domain priority:
- **Bureau (35%)** — historical repayment behavior is the strongest predictor of default.
- **Income (25%)** — capacity to repay is the second most important factor.
- **Cashflow (20%)** — bank statement behavior corroborates income claims.
- **Fraud (15%)** — fraud flags can override otherwise strong applications.
- **Policy (5%)** — compliance is a gate, not a discriminator; most applicants pass.

### 7.2 Decision Thresholds

| Score Range | Recommendation | Decision Authority |
|---|---|---|
| ≥ 750 | AUTO_APPROVE | System (no human review) |
| 650–749 | APPROVE_WITH_CONDITIONS | Branch Credit Manager |
| 550–649 | REFER_TO_CREDIT_COMMITTEE | Regional Credit Head |
| < 550 | AUTO_REJECT | System (no human review) |
| Policy ineligible | AUTO_REJECT | System override regardless of score |

### 7.3 Two-Layer Decision

The decision agent runs two layers:
1. **Rule layer** — composite score + `_score_to_recommendation()`. Deterministic, instant.
2. **LLM layer** — the LLM sees the rule-based recommendation as a hint but generates its own recommendation, rationale, conditions, strengths, and risks.

The LLM can upgrade or downgrade the rule recommendation. For example, a score of 762 (AUTO_APPROVE) might be downgraded to APPROVE_WITH_CONDITIONS if the LLM notices that the applicant's existing EMI burden is very close to the FOIR limit.

---

## 8. LLM Layer

### 8.1 Model Assignment

| Agent | Model | Reason |
|---|---|---|
| bureau_agent | `gemini-2.5-flash-lite` | Short narrative, low latency, high quota |
| income_agent | `gemini-2.5-flash-lite` | Same |
| bank_agent | `gemini-2.5-flash-lite` | Same |
| fraud_agent | `gemini-2.5-flash-lite` | Same |
| policy_rag_agent | `gemini-2.5-flash-lite` | Structured output parsing |
| decision_agent | `gemini-2.5-flash` | Needs stronger reasoning for complex credit judgment |
| report_agent | `gemini-2.5-flash` | Longer narrative, higher quality required |

### 8.2 LLM Factory (`utils/llm.py`)

```python
_rate_limiter = InMemoryRateLimiter(requests_per_second=0.2, check_every_n_seconds=0.5)

@lru_cache(maxsize=1)
def get_fast_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL_FAST,
        google_api_key=config.GOOGLE_API_KEY,
        max_output_tokens=512,
        temperature=0.1,       # slight creativity for summaries
        rate_limiter=_rate_limiter,
    )

@lru_cache(maxsize=1)
def get_smart_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL_SMART,
        google_api_key=config.GOOGLE_API_KEY,
        max_output_tokens=2048,
        temperature=0.0,       # deterministic decisions
        rate_limiter=_rate_limiter,
    )
```

**`@lru_cache`:** Both functions are cached. The first call initializes the LLM connection; all subsequent calls (including from parallel agents) return the same instance. Prevents redundant SDK initialization.

**`InMemoryRateLimiter`:** Limits all LLM calls to 0.2 requests/second (12 per minute), below the free-tier RPM limit of 15/minute. Without this, the five parallel analysis agents plus the decision and report agents would collectively fire 7 requests nearly simultaneously and hit rate limit errors.

**`temperature=0.0` for decision agent:** The credit decision must be reproducible and non-random. A temperature of 0 makes the LLM deterministic for the same input.

### 8.3 LLM Prompt Design

All prompts use `ChatPromptTemplate.from_messages()` with a system message and a human message. System messages set the persona and constraints; human messages provide the data.

Analysis agent prompts are tightly constrained: "2–3 sentences," "cite specific numbers," "no preamble." This prevents the LLM from producing verbose output that inflates latency and costs tokens.

Decision agent and report agent prompts are more open-ended but still structured: the decision prompt specifies the exact Pydantic schema fields to fill; the report prompt specifies the four paragraph topics.

---

## 9. Entry Points: CLI, API, UI

### 9.1 CLI (`main.py`)

```bash
python main.py --segment prime         # generate + run a synthetic prime applicant
python main.py --file APP_001.json     # run a saved application
```

Uses `workflow.stream()` to print live agent trace as agents complete, then `workflow.invoke()` for the final result. The live trace shows the LangGraph stream events in real time.

### 9.2 REST API (`api/main.py`)

**Endpoints:**
- `POST /underwrite` — single application, returns full `UnderwriteResponse`
- `POST /underwrite/batch` — list of applications, processes sequentially
- `GET /health` — health check
- `GET /applications/{id}` — retrieve cached result (in-memory dict)

**Request validation:** FastAPI uses Pydantic models (`ApplicantProfile`, `BureauReport`, `Transaction`) for request validation. Age must be 18–70, CIBIL must be 300–900, income/EMI must be positive — all validated before the workflow runs.

**In-memory cache:** Results are stored in `_results: dict[str, dict]` keyed by `application_id`. In a production system this would be Redis or a database.

Run with:
```bash
uvicorn api.main:api --reload --port 8000
```

### 9.3 Streamlit UI (`ui/app.py`)

**Layout:**
- Left column: loan application form (applicant details, loan request, bureau/CIBIL inputs)
- Right column: live agent trace (updates as each agent completes)
- Below the fold: decision banner, score metrics, 3-tab result view

**Live agent trace:** The UI calls `workflow.stream(state, stream_mode="updates")` and processes events in a loop. Each event carries the node name and output. The `render_trace()` function shows completed agents with checkmarks, the current running agent with a spinner, and pending agents with empty boxes.

**Three result tabs:**
1. **Credit Memo** — full markdown memo rendered by Streamlit
2. **Score Breakdown** — Plotly radar chart + score table + all flags
3. **Full Trace** — raw `agent_trace` log lines

**Sidebar sample loaders:**
- "Load Prime Applicant" — pre-fills the form with Priya Sharma, CIBIL 778, income ₹95k
- "Load Subprime Applicant" — pre-fills Ramesh Kumar, CIBIL 548, income ₹28k, written-off True
- "Load Random Synthetic" — picks a random file from `data/synthetic/`

---

## 10. Evaluation Suite

### 10.1 eval_runner.py

The batch runner:
1. Loads all JSON files from the labeled eval set directory.
2. Constructs `initial_state()` for each and calls `workflow.invoke()`.
3. Collects predicted recommendation, credit score, confidence, FOIR, latency, and errors.
4. Passes results to `summary_report()`.
5. Prints a formatted report and saves `_eval_results.json`.

**Binary bad mapping for Gini/KS:**
```python
_BAD_MAP = {
    "AUTO_REJECT":               1,   # bad
    "REFER_TO_CREDIT_COMMITTEE": 1,   # treat as bad (borderline)
    "APPROVE_WITH_CONDITIONS":   0,   # good
    "AUTO_APPROVE":              0,   # good
}
```

### 10.2 Metrics (`evaluation/metrics.py`)

**Gini Coefficient:**
```python
def gini_coefficient(y_true, scores):
    auc = roc_auc_score(y_true, -scores)   # invert: higher score = lower default
    return 2 * auc - 1
```
Measures discriminatory power. Range: −1 to +1. Target >0.45 for a retail credit model. Gini = 0 means the model is no better than random. Gini = 1 means perfect separation.

**KS Statistic (Kolmogorov-Smirnov):**
```python
def ks_statistic(y_true, scores):
    all_scores = sorted unique scores
    cdf_good = CDF of scores for y=0 (approved/good)
    cdf_bad  = CDF of scores for y=1 (rejected/bad)
    return max(abs(cdf_good - cdf_bad))
```
Maximum separation between the score distributions of good and bad applicants. Target >0.35. If KS is low, good and bad applicants score similarly — the model is not discriminating.

**Approval Rate:**
Fraction of cases with recommendation `AUTO_APPROVE` or `APPROVE_WITH_CONDITIONS`. A healthy rate for a personal loan portfolio is 55–70%.

**Auto-Decision Rate:**
Fraction of cases decided by the system without human review (`AUTO_APPROVE` or `AUTO_REJECT`). The rest require `APPROVE_WITH_CONDITIONS` (branch manager) or `REFER_TO_CREDIT_COMMITTEE` (regional head). Higher auto-decision rate = lower operational cost.

**FOIR Compliance Rate:**
```python
compliant = [f for rec, f in zip(recommendations, foiars)
             if rec in ("AUTO_APPROVE", "APPROVE_WITH_CONDITIONS") and f <= 0.50]
foir_compliance = compliant / total_approved
```
Checks that the system is not approving FOIR-violating cases. A value < 1.0 means the system has approved cases above the regulatory FOIR limit.

**Decision Agreement Rate:**
How often the model's recommendation matches the ground-truth label. This measures calibration, not just discrimination.

**Bad Rate by Score Band:**
```python
bands = [(0, 550), (550, 650), (650, 750), (750, 1001)]
```
For a well-calibrated model, bad rate should decrease monotonically as score increases. If the 750+ band has a higher bad rate than 650–750, the score is not rank-ordering risk correctly.

### 10.3 Observed Results (10-case smoke test)

| Metric | Value | Target |
|---|---|---|
| Error rate | 0.0 | 0.0 |
| FOIR compliance | 1.0 | 1.0 |
| Decision agreement | 0.4 | >0.7 |
| Approval rate | 1.0 | ~0.6 |
| Gini | NaN | >0.45 |
| KS | NaN | >0.35 |

Decision agreement of 40% and approval rate of 100% indicate the composite scoring is too lenient — the system approves most cases that the rule-based ground truth would reject. Root cause: sub-scores (especially cashflow and policy) default to high values for typical synthetic data. Score calibration is the next major work item.

---

## 11. Policy Documents

### `internal_credit_policy.md`

Simulates an NBFC's internal credit policy (FinEdge Capital Pvt. Ltd., version 2.3). Eight sections:

1. **Eligibility** — age (21–60), employment types (salaried, self-employed, professional, government), income minimums (₹20k Tier-1, ₹15k Tier-2), CIBIL minimum (550 hard floor, 650 preferred)
2. **FOIR Limits** — segment-specific caps (salaried ≥₹50k: 55%; self-employed: 45%). >60% = ineligible regardless of CIBIL.
3. **LTI Limits** — income-band-based multipliers (10× for ₹20–40k income; 18× for >₹80k). Product caps: personal ₹25L, MSME ₹1Cr, education ₹10L unsecured.
4. **Bank Statement Requirements** — 6 months minimum, ABB ≥ 1.5× EMI, bounce rate ≤ 3%, salary credited electronically in ≥5 of last 6 months.
5. **Derogatory Marks** — DPD-90+ or written-off = automatic rejection (no exception). DPD-30 >2 instances or >5 enquiries = refer to credit officer.
6. **Fraud Prevention** — velocity limits (>3 applications in 30 days = flag), income verification (stated vs. estimated within 20%), address consistency (PIN must match declared state).
7. **Interest Rate Bands** — risk-based pricing: 10.99–13.99% for 750+ scores, up to 19–24% for 550–649.
8. **Decision Authority** — maps score bands to system vs. branch manager vs. regional head.

### `rbi_fair_practice_guidelines.md`

Based on RBI NBFC Fair Practice Code requirements: transparent pricing disclosure, non-discrimination, customer grievance handling, prepayment rights, and collection practices.

---

## 12. Configuration and Constants

All system constants are in `config.py`. No magic numbers anywhere else.

```python
class Config:
    # LLM
    GOOGLE_API_KEY: str          # from .env
    GOOGLE_MODEL_FAST: str = "gemini-2.5-flash-lite"
    GOOGLE_MODEL_SMART: str = "gemini-2.5-flash"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Score weights (must sum to 1.0)
    BUREAU_WEIGHT: float = 0.35
    INCOME_WEIGHT: float = 0.25
    CASHFLOW_WEIGHT: float = 0.20
    FRAUD_WEIGHT: float = 0.15
    POLICY_WEIGHT: float = 0.05

    # Decision thresholds (0–1000 scale)
    AUTO_APPROVE_THRESHOLD: int = 750
    APPROVE_THRESHOLD: int = 650
    REFER_THRESHOLD: int = 550

    # Credit policy limits
    MAX_FOIR: float = 0.50
    MIN_CIBIL: int = 550
    MAX_LOAN_TO_INCOME_RATIO: float = 18   # 18× annual monthly income
    MIN_AGE: int = 21
    MAX_AGE: int = 60

    # Fraud thresholds
    CASH_WITHDRAWAL_RATIO_LIMIT: float = 0.40
    ENQUIRY_VELOCITY_LIMIT: int = 3
    INCOME_DISCREPANCY_PCT_LIMIT: float = 0.20

    # LLM temperature
    TEMPERATURE_NARRATIVE: float = 0.1
    TEMPERATURE_DECISION: float = 0.0
```

---

## 13. Test Suite

Three test files, 26 tests total. All pass.

### `tests/test_llm.py` (7 tests)

- `test_config_has_google_api_key` — GOOGLE_API_KEY is non-empty
- `test_config_models_set` — model names are non-empty strings
- `test_fast_llm_creates` — `get_fast_llm()` returns a `ChatGoogleGenerativeAI` with the correct model name
- `test_smart_llm_creates` — same for smart LLM
- `test_fast_llm_responds` — live API call returns non-empty string
- `test_smart_llm_responds` — same
- `test_smart_llm_structured_output` — `with_structured_output(SomeModel, method="function_calling")` parses correctly

### `tests/test_decision_agent.py` (13 tests)

Pure unit tests (no LLM, no network):
- `test_composite_score_all_perfect` — all sub-scores = 1000 → composite = 1000
- `test_composite_score_all_zero` — all zeros → composite = 0 (or low due to fraud inversion)
- `test_composite_score_weights_sum` — weights 0.35+0.25+0.20+0.15+0.05 = 1.0
- `test_fraud_score_inverted` — fraud_score=0 contributes 1000×0.15=150; fraud_score=1000 contributes 0
- `test_score_to_recommendation[800-AUTO_APPROVE]` — threshold mapping
- `test_score_to_recommendation[900-False-AUTO_REJECT]` — policy ineligible overrides score
- `test_decision_node_returns_required_fields` — integration test: decision node returns all expected keys
- `test_decision_node_score_in_range` — 0 ≤ credit_score ≤ 1000
- `test_decision_node_recommendation_valid` — recommendation is one of the 4 valid strings
- `test_decision_node_llm_not_fallback` — "Rule-based decision" NOT in rationale (confirms LLM ran)

### `tests/test_report_agent.py` (6 tests)

- `test_score_breakdown_table` — output contains all 5 component names
- `test_report_node_returns_memo` — node returns a `credit_memo` key
- `test_memo_contains_required_sections` — memo has all 6 required section headers
- `test_memo_contains_applicant_details` — applicant name appears in memo
- `test_narrative_is_llm_generated` — narrative section does not contain "narrative unavailable" (LLM ran)
- `test_agent_trace_appended` — agent_trace list is non-empty and contains "Report Agent"

Run with:
```bash
.venv/bin/pytest tests/ -v
```

---

## 14. What Can Be Improved

### 14.1 Score Calibration (Most Urgent)

**Current state:** Decision agreement 40%, approval rate near 100% on the eval set. The system is too lenient.

**Root cause:** Sub-score components have overly generous base values. The cashflow agent gives 300 base points just for not having a negative cashflow. The policy agent defaults to 800/1000 even when the vectorstore is unavailable. Default scores when data is missing are set to 500 (neutral), which still pushes most cases above the approval threshold.

**What to do:**
- Audit each agent's scoring formula. Lower base/floor values.
- Use the 200-case labeled eval set to calibrate thresholds: find the score cutoffs where Gini and KS are maximized.
- Consider a calibration layer (Platt scaling or isotonic regression) on top of the composite score to map raw scores to well-calibrated probabilities.

### 14.2 XGBoost Income Model

**Current state:** `income_agent.py` uses rule-based multipliers as a proxy for model inference.

**What to do:**
1. Download the [Home Credit Default Risk dataset](https://kaggle.com/c/home-credit-default-risk) from Kaggle.
2. Train an XGBoost model on `AMT_INCOME_TOTAL` and related features to predict income category.
3. Save as `models/income_xgb.pkl`.
4. Replace `_estimate_income()` in `income_agent.py` with `model.predict([features])[0]`.

This is already architecturally prepared — the function signature is unchanged; only the internals swap.

### 14.3 Real Data Calibration

**Current state:** Synthetic data distributions are hardcoded approximations.

**What to do:**
- **CIBIL distribution:** Use actual Indian credit bureau distribution data (or the Indian Credit Scoring Kaggle dataset) to set realistic mean/std for each segment.
- **FOIR distribution:** Use Lending Club or Home Credit data to calibrate the existing EMI ratio bands.
- **DPD ground truth:** Map `SeriousDlqin2yrs` from the Give Me Some Credit dataset to expected decision bands for calibrating the eval set labels.

### 14.4 Streaming API

**Current state:** `POST /underwrite` blocks until the full workflow completes (~30–60s). The client has no progress indication.

**What to do:**
- Add `POST /underwrite/stream` that returns a Server-Sent Events (SSE) stream.
- Use `workflow.stream(state, stream_mode="updates")` and yield each agent completion event.
- The client sees progress in real time, matching the Streamlit UI behavior.

### 14.5 Human-in-the-Loop for REFER Cases

**Current state:** `REFER_TO_CREDIT_COMMITTEE` cases are returned to the caller with no workflow path forward.

**What to do:**
- Add a `HumanInterrupt` node in LangGraph after `decision_agent` for REFER cases.
- Use LangGraph's built-in interrupt mechanism (`interrupt_before=["human_review"]`).
- The API returns a 202 Accepted with a review URL; a credit officer reviews and submits a decision; the workflow resumes.

This would make the system production-realistic for borderline cases.

### 14.6 Persistent Storage

**Current state:** API results are cached in `_results: dict[str, dict]` (in-memory, lost on restart).

**What to do:**
- Replace with PostgreSQL (application records) + Redis (result cache with TTL).
- Use SQLAlchemy for the application audit trail.
- Each application, all agent outputs, decision, memo, and latency stored for regulatory audit.

### 14.7 Adverse Action Notices

**Current state:** Rejected applicants receive a recommendation string and rationale, but no formal regulatory notice.

**What to do:**
- Add an `adverse_action_agent` that runs only for `AUTO_REJECT` and `REFER_TO_CREDIT_COMMITTEE` cases.
- Generate a formal adverse action notice citing specific reasons (required by RBI fair practice guidelines).
- The report agent should render this as a separate section of the credit memo.

### 14.8 Dockerization

**Current state:** No Docker setup. Requires manual environment setup with `.venv`.

**What to do:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:api", "--host", "0.0.0.0", "--port", "8000"]
```

A `docker-compose.yml` would bring up the API, UI, and ChromaDB persistence volume together.

### 14.9 Observability

**Current state:** All observability is in `agent_trace` (a list of strings). No metrics, no structured logging, no traces.

**What to do:**
- Add LangSmith tracing (already a dependency in `langsmith>=0.8.5`): set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`.
- Emit Prometheus metrics: per-agent latency histograms, decision distribution counters, error rates.
- Structured JSON logging to replace print statements.

### 14.10 Policy Document Refresh

**Current state:** Policy docs are static Markdown files. ChromaDB vectorstore must be manually rebuilt when policies change.

**What to do:**
- Add a `POST /admin/reload-policies` endpoint that clears the `@lru_cache` and deletes the ChromaDB persist directory.
- On next request, the vectorstore auto-rebuilds from the updated documents.
- Alternatively, watch the `policy_docs/` directory for changes and trigger rebuild automatically.

---

*Generated by CreditIQ Technical Documentation — 2026-05-23*
