# CreditIQ — Development Log

Living document. Updated each session. Tracks features implemented, design decisions, open questions, and planned work.

---

## Project Snapshot

**Goal:** LangGraph multi-agent system that autonomously underwrites Indian personal/MSME loan applications — built as a portfolio project for EF/MLE interviews (Albert, SMFG India Credit background).

**Status:** Full implementation complete. **End-to-end confirmed running as of 2026-05-23.**

---

## File Map

```
creditUnderwritingAgent/
├── agents/
│   ├── bureau_agent.py          ✅ Rules + LLM narrative
│   ├── income_agent.py          ✅ Rules + LLM | XGBoost stub (not wired)
│   ├── bank_statement_agent.py  ✅ Pandas pipeline + LLM
│   ├── fraud_agent.py           ✅ Rules engine + LLM
│   ├── policy_rag_agent.py      ✅ ChromaDB RAG + LLM
│   ├── decision_agent.py        ✅ Pydantic structured output (Claude Sonnet)
│   └── report_agent.py          ✅ Full markdown credit memo
├── graph/
│   ├── state.py                 ✅ UnderwritingState TypedDict + initial_state()
│   └── workflow.py              ✅ LangGraph StateGraph (fan-out/fan-in)
├── data/
│   ├── policy_docs/
│   │   ├── internal_credit_policy.md     ✅ Fake internal NBFC policy (8 sections)
│   │   └── rbi_fair_practice_guidelines.md ✅ RBI-based guidelines
│   ├── synthetic/               ⬜ Generated at runtime (run generate_synthetic_data.py)
│   └── eval_set/                ⬜ Generated at runtime (--labeled flag)
├── models/
│   └── income_xgb.pkl           ⬜ Not yet trained (income_agent uses rule-based fallback)
├── vectorstore/
│   └── policy_chroma/           ⬜ Built at runtime (lazy init or setup_vectorstore.py)
├── api/
│   └── main.py                  ✅ FastAPI: POST /underwrite, /underwrite/batch, GET /health
├── ui/
│   └── app.py                   ✅ Streamlit: form + live agent trace + radar chart + tabs
├── evaluation/
│   ├── eval_runner.py           ✅ Batch eval, latency stats, saves _eval_results.json
│   └── metrics.py               ✅ Gini, KS, approval rate, bad-rate-by-band
├── scripts/
│   ├── generate_synthetic_data.py ✅ Faker + NumPy, 3 segments, optional --labeled
│   └── setup_vectorstore.py     ✅ Pre-warms ChromaDB from policy_docs/
├── utils/
│   └── llm.py                   ✅ LLM factory (@lru_cache singletons)
├── config.py                    ✅ All constants, weights, thresholds
├── main.py                      ✅ CLI entry point
├── requirements.txt             ✅ All dependencies pinned
├── .env.example                 ✅ ANTHROPIC_API_KEY template
└── README.md                    ✅ Quickstart guide
```

---

## Architecture

### Graph Topology

```
START
  └─► timing_start
            ├─► bureau_agent   ─┐
            ├─► income_agent   ─┤
            ├─► bank_agent     ─┼─► decision_agent ─► report_agent ─► timing_end ─► END
            ├─► fraud_agent    ─┤
            └─► policy_agent   ─┘
               (all 5 in parallel)
```

Five agents run in parallel (LangGraph fan-out). `decision_agent` waits for all five (fan-in), then `report_agent` generates the credit memo.

### Composite Score Formula

```
Credit Score (0–1000) =
    0.35 × bureau_score
  + 0.25 × income_score
  + 0.20 × cashflow_score
  + 0.15 × (1000 − fraud_score)   ← inverted: low fraud = high contribution
  + 0.05 × policy_score
```

### Decision Thresholds

| Score     | Recommendation                  |
|-----------|---------------------------------|
| ≥ 750     | AUTO_APPROVE                    |
| 650–749   | APPROVE_WITH_CONDITIONS         |
| 550–649   | REFER_TO_CREDIT_COMMITTEE       |
| < 550     | AUTO_REJECT                     |

---

## Data Flow

Every input — whether from the synthetic generator, Streamlit form, REST API, or a saved JSON file — is normalized into the same `UnderwritingState` dict before the graph runs. Agents never know the data's origin.

```
┌─── SOURCE (any one) ────────────────────────────────────────┐
│  Synthetic generator   │  Streamlit form  │  API POST body  │
│  (JSON files)          │  (user input)    │  (/underwrite)  │
└────────────────────────┴──────────────────┴─────────────────┘
                                │
                     initial_state() factory
                                │
              UnderwritingState dict (typed, all optional fields)
                                │
                          LangGraph graph
                                │
                        ┌───────┴────────┐
                   stream events       invoke()
                   (UI live trace)   (API/CLI result)
```

---

## LLM Usage

| Component         | Model                      | Purpose                              |
|-------------------|----------------------------|--------------------------------------|
| bureau_agent      | gemini-2.0-flash           | Narrative summary of bureau analysis |
| income_agent      | gemini-2.0-flash           | Income discrepancy narrative         |
| bank_agent        | gemini-2.0-flash           | Cashflow narrative                   |
| fraud_agent       | gemini-2.0-flash           | Fraud flag narrative                 |
| policy_rag_agent  | gemini-2.0-flash           | Policy compliance check              |
| decision_agent    | gemini-2.5-flash           | Structured credit decision (Pydantic)|
| report_agent      | gemini-2.5-flash           | Full credit memo generation          |

All models via Google AI Studio (`GOOGLE_API_KEY`).

Free tier quotas (as observed):
- `gemini-2.5-pro`: 0 req/day free tier — **unusable**
- `gemini-2.5-flash`: 20 req/day — exhausted in one session, too low for any real use
- `gemini-2.0-flash`: 200 req/day — current choice for both fast + smart models

All LLM calls wrapped in try/except with rule-based fallback — system works even if LLM is unavailable.

---

## Data Sources

### What's Implemented (Synthetic)

All runtime data is synthetic. The generator (`scripts/generate_synthetic_data.py`) uses Faker + NumPy with domain-specific distributions for 3 segments:

| Segment    | CIBIL Range | Income Range      | DPD Likelihood |
|------------|-------------|-------------------|----------------|
| prime      | ~750±40     | ₹80k–₹2L/month   | Very low       |
| near_prime | ~680±50     | ₹40k–₹1L/month   | Low            |
| subprime   | ~580±80     | ₹20k–₹60k/month  | High           |

### Kaggle Datasets (Referenced in Plan — Not Yet Integrated)

The plan mentions these for calibration and training. They don't change the runtime data flow — they're offline tools.

| Dataset                     | Kaggle URL                                  | Integration Point                                           |
|-----------------------------|---------------------------------------------|-------------------------------------------------------------|
| Home Credit Default Risk    | `kaggle.com/c/home-credit-default-risk`     | Calibrate synthetic distribution params; train XGBoost model |
| Give Me Some Credit         | `kaggle.com/c/GiveMeSomeCredit`             | DPD/delinquency model; eval set ground-truth labels         |
| Lending Club Loan Data      | `kaggle.com/datasets/wordsforthewise/lending-club` | Income agent DTI calibration                         |
| UCI Credit Card Default     | UCI ML Repo                                 | Score validation baseline                                   |
| Indian Credit Scoring       | `kaggle.com/search?q=india+credit+score`    | India-specific CIBIL distribution params                    |

**How they'd plug in:**
1. **Distribution calibration** — compute real mean/std for CIBIL by approval status, replace hardcoded values in `generate_synthetic_data.py`
2. **XGBoost income model** — train on Home Credit `AMT_INCOME_TOTAL` features → save `models/income_xgb.pkl` → `income_agent.py` loads it instead of rule-based estimate
3. **Eval ground truth** — map `SeriousDlqin2yrs` from Give Me Some Credit to expected decision bands for the 200-case labeled eval set

---

## Feature Status

### Implemented

- [x] LangGraph StateGraph with parallel fan-out/fan-in
- [x] Bureau Analysis Agent — CIBIL parsing, FOIR, DPD, utilization → `bureau_score/1000`
- [x] Income Verification Agent — employment multipliers, city-tier adjustments, salary stability → `income_score/1000`
- [x] Bank Statement Agent — ABB, bounce rate, cash withdrawal ratio → `cashflow_score/1000`
- [x] Fraud Detection Agent — velocity, cash ratio, income-EMI consistency, PIN-state mapping → `fraud_score/1000`
- [x] Policy RAG Agent — ChromaDB over 2 policy docs, lazy init, top-4 retrieval → `policy_score/1000`
- [x] Decision Agent — composite score + Claude Sonnet structured output (Pydantic)
- [x] Report Agent — full markdown credit memo with score breakdown table + LLM narrative
- [x] Synthetic data generator (3 segments, optional ground-truth labels)
- [x] Evaluation suite — Gini coefficient, KS statistic, approval rate, bad-rate-by-band
- [x] FastAPI REST endpoints
- [x] Streamlit UI — live agent trace, radar chart, 3-tab results
- [x] CLI entry point
- [x] ChromaDB vectorstore setup script
- [x] 2 policy documents (internal credit policy + RBI fair practice guidelines)

### Not Yet Done / Stubs

- [ ] XGBoost income model (`models/income_xgb.pkl`) — income_agent uses rule-based estimate
- [ ] Kaggle dataset integration for distribution calibration
- [ ] Docker / docker-compose
- [ ] Snowflake audit logging (stretch goal from plan)
- [ ] Human-in-the-loop refer flow (stretch goal)
- [ ] Adverse action notice generation (stretch goal)
- [ ] End-to-end run (dependencies not yet installed)

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| LLM provider | Anthropic API direct (not Bedrock) | Simpler setup, no AWS credentials, same Claude models |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local, ~80MB, no API key, good quality for policy RAG |
| State merging | `Annotated[List[str], operator.add]` for `agent_trace` and `errors` | Lets parallel agents append without conflict |
| LLM singletons | `@lru_cache` in `utils/llm.py` | One connection per process, avoids re-init overhead |
| Structured output | `llm.with_structured_output(CreditDecision)` | Forces decision agent into constrained Pydantic schema |
| ChromaDB init | Lazy on first `policy_rag_node` call | No startup delay; `setup_vectorstore.py` allows pre-warming |
| TypedDict | `total=False` with `initial_state()` factory | All fields optional at type level; factory sets required ones |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. Generate synthetic data
python scripts/generate_synthetic_data.py --count 500 --out data/synthetic

# 4. (Optional) Pre-warm vectorstore
python scripts/setup_vectorstore.py

# 5. CLI demo
python main.py --segment prime
python main.py --segment subprime
python main.py --file data/synthetic/APP_202501_0001.json

# 6. Streamlit UI
streamlit run ui/app.py

# 7. REST API
uvicorn api.main:api --reload --port 8000

# 8. Evaluation (after generating labeled eval set)
python scripts/generate_synthetic_data.py --count 200 --out data/eval_set --labeled
python evaluation/eval_runner.py --eval-dir data/eval_set
python evaluation/eval_runner.py --eval-dir data/eval_set --sample 10  # quick smoke test
```

---

## Session Log

### Session 2 — 2026-05-23
**Switched LLM provider: Anthropic → Google AI Studio**
- `gemini-2.5-pro` attempted first — quota=0 on free tier, decision/report agents fell back to rule-based
- Switched smart model to `gemini-2.5-flash` (free tier) — LLM decision + report now fully working
- Fast model stays `gemini-2.0-flash` for analysis agents
- `langchain-google-genai` installed; `langchain-anthropic` removed from requirements
- `.env` key changed from `ANTHROPIC_API_KEY` to `GOOGLE_API_KEY`

**Bug fixes:**
- Loan amount formula was `income × 12 × (12-18)` = 144-216× monthly income — way too high
- Fixed to `income × (12-30)` for prime, `(6-18)` near_prime, `(3-10)` subprime
- Removed unused `os` import and prefixed unused `segment` param in `generate_synthetic_data.py`

**First successful end-to-end run:**
- All 5 parallel agents completed; decision + report generated by LLM
- Sample: Wazir Chhabra | ₹2L/month | ₹50L loan ask | CIBIL 700
- Result: APPROVE_WITH_CONDITIONS, score 738/1000, confidence 90%
- LLM produced specific conditions: "reduce loan to ₹22.64L for FOIR compliance"
- Latency: ~51s total (35s for parallel agents — free tier rate limiting)

**Eval smoke-test results (10 cases):**
- Error rate: 0.0 — all cases ran cleanly
- FOIR compliance: 1.0 — every approved case had valid FOIR
- Decision agreement: 0.4 — system agrees with ground-truth labels 40% of the time
- Approval rate: 1.0 — system approved all 10 (expected ~58% based on label distribution)
- Root cause: composite score running too high — bureau/cashflow sub-scores push most cases above 750 threshold
- Gini/KS: NaN — all cases scored ≥650, no variance to compute discrimination stats
- **Next action needed: calibrate score component weights (Phase 4 tuning)**
- Latency: ~106s avg per application (free tier rate limiting — was 74s on first run, varies)

**Bug fix — decision agent narrative corruption (2026-05-23):**
- `gemini-2.5-flash` thinking mode leaked as `\n\n\n\n...` characters into JSON string fields
  when using default JSON mode in `with_structured_output`
- This caused Pydantic to fail parsing `CreditDecision` (missing `decision_rationale` field)
- Decision agent silently fell back to rule-based; report narrative showed raw error
- Fix: `with_structured_output(CreditDecision, method="function_calling")` — function calling
  mode enforces strict schema and keeps thinking tokens out of the output
- Also fixed: `datetime.utcnow()` → `datetime.now(timezone.utc)` deprecation in report_agent

**Tests added (tests/):**
- `test_llm.py` — 7 tests: API key present, model names set, LLM creates, responds, structured output works
- `test_decision_agent.py` — 13 tests: composite scoring unit tests (pure), recommendation thresholds,
  node integration (LLM), structured output not falling back
- `test_report_agent.py` — 6 tests: score table, memo structure, required sections, narrative is LLM-generated
- Run: `.venv/bin/pytest tests/ -v` → **26/26 passed**

**Known limitations / open items:**
- Score calibration too lenient — agreement 40%, needs weight tuning (Phase 4)
- Latency ~30-50s per LLM call with billing enabled (much better than free tier)
- `models/income_xgb.pkl` not yet trained — income agent uses rule-based estimate
- FastAPI not yet run/tested

**Environment:**
- Python 3.12 via `.venv/` (system pip not available; venv has pip)
- `.vscode/settings.json` created pointing to `.venv/bin/python` (select it with Ctrl+Shift+P → Python: Select Interpreter)

### Session 3 — 2026-05-23
**Generated PROJECT_REPORT.md** — comprehensive technical document covering:
- Every agent's scoring formula and flags (bureau, income, bank, fraud, policy)
- LangGraph state design, reducer annotations, fan-out/fan-in mechanics
- FOIR-aware synthetic data generation design
- Full RAG pipeline (chunking params, embeddings, ChromaDB lazy init)
- Composite scoring formula rationale
- LLM layer: model assignment, rate limiter, function_calling fix
- All entry points (CLI, FastAPI, Streamlit)
- Evaluation metrics (Gini, KS, approval rate, FOIR compliance, decision agreement)
- Policy document coverage (8 sections)
- 10 concrete improvement areas (score calibration, XGBoost, streaming API, HITL, Docker, etc.)

No code changes this session — documentation only.

---

### Session 1 — 2026-05-23
- Implemented full system in one session (all 4 phases from plan)
- Created 29 files across all directories
- LLM provider chosen: Anthropic API direct (later changed to Google AI Studio in session 2)
- Embeddings: HuggingFace all-MiniLM-L6-v2 (local, no API key)
- All agents, graph, API, UI, eval, scripts, config written
- Clarified data flow: Kaggle datasets are offline calibration tools, not runtime inputs
