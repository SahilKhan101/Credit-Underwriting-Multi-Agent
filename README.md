# CreditIQ — Multi-Agent Credit Underwriting System

LangGraph-powered system that autonomously underwrites personal/MSME loans.
Five specialized agents run in parallel → Decision Agent → structured credit memo.

## Architecture

```
START → timing_start
            ├── bureau_agent   ─┐
            ├── income_agent   ─┤
            ├── bank_agent     ─┼──► decision_agent ──► report_agent ──► END
            ├── fraud_agent    ─┤
            └── policy_agent   ─┘
```

| Agent | Role |
|---|---|
| Bureau | CIBIL parsing, FOIR, DPD, utilization → bureau_score/1000 |
| Income | Stated vs. estimated income, stability → income_score/1000 |
| Bank Statement | ABB, bounce rate, salary regularity → cashflow_score/1000 |
| Fraud | Velocity, cash withdrawal, address/income consistency → fraud_score/1000 |
| Policy RAG | ChromaDB over RBI/internal policy docs → policy_score/1000 |
| Decision | Composite score + Claude Sonnet structured output |
| Report | Full markdown credit memo |

**Composite Score** = 0.35×Bureau + 0.25×Income + 0.20×Cashflow + 0.15×(1000-Fraud) + 0.05×Policy

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. Generate synthetic data (500 applications)
python scripts/generate_synthetic_data.py --count 500 --out data/synthetic

# 4. (Optional) Pre-warm ChromaDB vectorstore
python scripts/setup_vectorstore.py

# 5. Run CLI demo
python main.py --segment prime

# 6. Streamlit UI
streamlit run ui/app.py

# 7. REST API
uvicorn api.main:api --reload --port 8000
```

## Evaluation

```bash
# Generate labeled eval set
python scripts/generate_synthetic_data.py --count 200 --out data/eval_set --labeled

# Run evaluation (all 200 cases)
python evaluation/eval_runner.py --eval-dir data/eval_set

# Quick smoke test (10 cases)
python evaluation/eval_runner.py --eval-dir data/eval_set --sample 10
```

## Decision Thresholds

| Score | Recommendation |
|---|---|
| ≥ 750 | AUTO APPROVE |
| 650–749 | APPROVE WITH CONDITIONS |
| 550–649 | REFER TO CREDIT COMMITTEE |
| < 550 | AUTO REJECT |

## Tech Stack

- **LangGraph** — StateGraph orchestration, parallel fan-out/fan-in
- **LangChain + Claude** — Haiku (fast agents) + Sonnet (decision + report)
- **ChromaDB + HuggingFace** — Local RAG for policy documents (no extra API key)
- **FastAPI** — REST API
- **Streamlit + Plotly** — Live agent trace UI with radar chart
- **Pandas / NumPy / XGBoost** — Feature extraction pipeline
- **Faker** — Synthetic Indian loan application generator
