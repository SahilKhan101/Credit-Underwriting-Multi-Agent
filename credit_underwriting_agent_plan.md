# Multi-Agent Credit Underwriting Assistant
## Full Implementation Plan — Albert (SMFG India Credit Background)

---

## 1. PROJECT OVERVIEW

**What it is:** A LangGraph-powered multi-agent system that autonomously underwrites personal/MSME loan applications. It ingests a loan application → pulls and analyzes bureau data, income signals, bank statements, and fraud indicators → deliberates across specialized agents → produces a structured credit decision memo with risk score, approval recommendation, and explainability narrative.

**Why it's impressive:**
- End-to-end agentic system (not just a chatbot wrapper)
- Domain-authentic: mirrors real NBFC underwriting workflows
- Uses your exact stack (LangGraph, LangChain, AWS Bedrock, ChromaDB, Snowflake)
- Has real evaluation metrics — not just vibes
- Pitchable to EF, defensible in MLE/DS interviews

---

## 2. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│         (LangGraph StateGraph — central supervisor)          │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
  [Bureau]   [Income]   [Bank Stmt] [Fraud]   [Policy RAG]
   Agent      Agent       Agent      Agent      Agent
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                            │
                            ▼
                    DECISION AGENT
                  (Final credit memo)
                            │
                            ▼
                  REPORT GENERATION AGENT
              (PDF/JSON structured output)
```

### Agent Responsibilities

| Agent | Role | Tools |
|---|---|---|
| **Orchestrator** | Routes application across agents, manages state, handles retries | LangGraph StateGraph |
| **Bureau Analysis Agent** | Parses CIBIL-like report, computes derived features (DPD, utilization, age of credit) | Python parser + LLM summarizer |
| **Income Verification Agent** | Estimates income from ITR proxy / salary slip / GST returns, flags discrepancies | XGBoost income model + LLM |
| **Bank Statement Agent** | Analyzes 6-month statement: avg balance, salary credits, EMI obligations, cash flow volatility | Pandas + LLM narrative |
| **Fraud Detection Agent** | Flags address inconsistency, name mismatch, velocity checks, synthetic identity signals | Rules engine + LLM |
| **Policy RAG Agent** | Queries RBI/internal policy docs to validate if the case is eligible under current norms | ChromaDB + LangChain RAG |
| **Decision Agent** | Aggregates all agent outputs, computes final credit score, recommends Approve/Reject/Refer | LLM with structured output |
| **Report Agent** | Generates human-readable credit memo with reasoning, confidence scores, and exceptions | LLM + PDF/JSON formatter |

---

## 3. DATA SOURCES

### 3A. Training / Simulation Data (Free & Public)

| Dataset | Source | Use |
|---|---|---|
| **Home Credit Default Risk** | Kaggle | Bureau features, income, credit history. 300K+ rows. |
| **Give Me Some Credit** | Kaggle | Delinquency prediction, utilization, DPD features |
| **Lending Club Loan Data** | Kaggle | Loan purpose, DTI, grade, interest rate signals |
| **UCI Credit Card Default** | UCI ML Repo | Taiwan dataset — bureau + payment behavior |
| **Indian Credit Scoring Dataset** | Kaggle (search "India credit score") | India-specific features |
| **CIBIL Score Simulator** | Synthetic generation (you'll build this) | Simulated bureau reports in JSON |

### 3B. Synthetic Data Generation (Critical for Demo)

Since real CIBIL data is not public, **generate realistic bureau reports programmatically:**

```python
# Example: Synthetic CIBIL-like JSON
{
  "applicant_id": "APP_2024_001",
  "cibil_score": 712,
  "total_accounts": 4,
  "active_accounts": 2,
  "credit_utilization": 0.38,
  "dpd_30_count": 1,        # Days Past Due 30+ in last 12 months
  "dpd_90_count": 0,
  "enquiries_last_6m": 3,
  "oldest_account_months": 48,
  "current_balance": 245000,
  "sanctioned_limit": 650000,
  "secured_unsecured_ratio": 0.6,
  "written_off_flag": False
}
```

Use **Faker + domain-specific distributions** (e.g., CIBIL scores are roughly normal around 680-720 for approved population, skewed left for rejected).

### 3C. Bank Statement Simulation

Generate 6-month synthetic bank statements with:
- Monthly salary credits (with variability ±5%)
- EMI debits (fixed amounts on specific dates)
- Cash withdrawals (fraud signal if >40% of income)
- Minimum balance compliance
- Bounce indicators

### 3D. Policy Documents (for RAG Agent)

Real documents you can use:
- **RBI Master Direction on KYC** (rbi.org.in — publicly available PDF)
- **RBI Fair Practice Code for NBFCs** (public PDF)
- **RBI FLDG Guidelines 2023** (public)
- **Your own internal policy** — write a fake "Internal Credit Policy v2.3" doc (2-3 pages of rules like LTV caps, FOIR limits, eligible income types)

These go into ChromaDB as chunks for the Policy RAG Agent.

---

## 4. TECH STACK

```
LangGraph           — Agent orchestration, state management, conditional edges
LangChain           — LLM calls, tool wrappers, RAG chain
AWS Bedrock         — Claude 3 Sonnet/Haiku as LLM backbone
ChromaDB            — Vector store for policy documents
Snowflake (optional)— Store application data, decisions, audit logs
PostgreSQL/SQLite   — Local dev database
FastAPI             — REST API for the underwriting system
Streamlit           — Demo UI (loan application form → live agent trace → decision)
XGBoost             — Your existing income estimation model (plug it in as a tool)
Pandas              — Bank statement feature extraction
Pydantic            — Structured output schemas for each agent
Docker              — Containerize for demo
```

---

## 5. LANGGRAPH STATE DESIGN

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

class UnderwritingState(TypedDict):
    # Input
    application_id: str
    applicant_profile: dict          # Name, age, employment, loan ask

    # Agent outputs (populated progressively)
    bureau_analysis: dict            # Scores, flags, derived features
    income_analysis: dict            # Verified income, discrepancy flags
    bank_statement_analysis: dict    # Cash flow, EMI, bounce flags
    fraud_signals: dict              # Risk flags from fraud agent
    policy_check: dict               # Eligible/ineligible + reason

    # Decision
    credit_score: float              # 0-1000 composite score
    recommendation: str              # APPROVE / REJECT / REFER_TO_CREDIT_COMMITTEE
    confidence: float                # 0.0 - 1.0
    decision_rationale: str

    # Metadata
    agent_trace: Annotated[List[str], operator.add]   # Audit log
    errors: List[str]
    processing_time_ms: int
```

### Graph Flow

```python
graph = StateGraph(UnderwritingState)

graph.add_node("bureau_agent", bureau_analysis_node)
graph.add_node("income_agent", income_verification_node)
graph.add_node("bank_agent", bank_statement_node)
graph.add_node("fraud_agent", fraud_detection_node)
graph.add_node("policy_agent", policy_rag_node)
graph.add_node("decision_agent", credit_decision_node)
graph.add_node("report_agent", report_generation_node)

# Parallel execution for independent agents
graph.add_edge("bureau_agent", "decision_agent")
graph.add_edge("income_agent", "decision_agent")
graph.add_edge("bank_agent", "decision_agent")
graph.add_edge("fraud_agent", "decision_agent")
graph.add_edge("policy_agent", "decision_agent")
graph.add_edge("decision_agent", "report_agent")
graph.add_edge("report_agent", END)
```

---

## 6. CREDIT SCORING LOGIC

### Composite Score Formula (0–1000)

```
Credit Score = w1*Bureau_Score + w2*Income_Score + w3*Cashflow_Score 
               + w4*Fraud_Score + w5*Policy_Score

Weights (tunable):
  w1 = 0.35   # Bureau (CIBIL-like)
  w2 = 0.25   # Income adequacy
  w3 = 0.20   # Bank statement health
  w4 = 0.15   # Fraud risk (inverted — low fraud = high score)
  w5 = 0.05   # Policy compliance
```

### Decision Thresholds

| Score | Recommendation |
|---|---|
| ≥ 750 | AUTO APPROVE |
| 650–749 | APPROVE with conditions |
| 550–649 | REFER TO CREDIT COMMITTEE |
| < 550 | AUTO REJECT |

### Key Derived Features per Agent

**Bureau Agent:**
- FOIR (Fixed Obligation to Income Ratio) = existing EMIs / monthly income
- Credit Utilization = current balance / sanctioned limit
- Derogatory flag = any DPD 90+ or write-offs in last 24 months

**Income Agent:**
- Income stability score = stddev(last 6 months salary) / mean salary
- Income-to-loan ratio = annual income / loan ask
- Discrepancy flag = |stated income - estimated income| > 20%

**Bank Statement Agent:**
- ABB (Average Bank Balance) as % of EMI
- Bounce rate = bounced transactions / total debit transactions
- Salary regularity = variance in salary credit dates

**Fraud Agent:**
- Velocity flag = >3 loan enquiries in 30 days
- Synthetic identity score (name/DOB/address consistency)
- Cash withdrawal ratio = cash_out / total_credits

---

## 7. RAG SETUP (Policy Agent)

```python
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings

# Chunk policy PDFs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# Embed with Bedrock (Titan Embeddings)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="ap-south-1"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./policy_vectorstore"
)

# Query at runtime
def policy_check(application: dict) -> dict:
    query = f"Is a {application['loan_type']} loan of ₹{application['loan_amount']} 
              eligible for applicant aged {application['age']} with income 
              {application['income']}?"
    
    relevant_policies = vectorstore.similarity_search(query, k=4)
    # Pass to LLM for policy compliance check
```

---

## 8. EVALUATION METRICS

### 8A. Model-Level Metrics (Credit Decision Quality)

| Metric | Formula | Target | Why |
|---|---|---|---|
| **Approval Rate** | Approved / Total | ~65-75% (mirrors real NBFC) | Sanity check |
| **Auto Decision Rate** | Auto approve + auto reject / Total | >80% | Efficiency |
| **Gini Coefficient** | 2*AUC - 1 | >0.45 | Discriminatory power |
| **KS Statistic** | max(TPR - FPR) | >0.35 | Score separation |
| **Bad Rate @ Score Band** | Defaults in each score bucket | Monotone decreasing | Score calibration |
| **FOIR Compliance %** | Apps where FOIR < 50% are approved | 100% | Policy adherence |

### 8B. Agent-Level Metrics

| Agent | Metric | Target |
|---|---|---|
| Bureau Agent | Feature extraction accuracy (vs. ground truth) | >95% |
| Income Agent | Income estimation MAPE | <15% |
| Fraud Agent | Precision on known fraud cases | >85% |
| Policy Agent | Retrieval accuracy (relevant chunk in top-3) | >90% |
| Decision Agent | Decision consistency (same input → same output) | >98% |

### 8C. System-Level Metrics

| Metric | Target |
|---|---|
| End-to-end latency | < 30 seconds per application |
| Agent retry rate | < 5% |
| Hallucination rate (LLM outputs) | < 2% (measure with eval set) |
| Structured output parse success | > 99% |
| Audit trail completeness | 100% (every decision traceable) |

### 8D. Evaluation Dataset Design

Create a **labeled eval set of 200 synthetic applications:**
- 100 clear approvals (high CIBIL, stable income, low DPD)
- 50 clear rejections (low score, DPD 90+, income mismatch)
- 50 borderline (refer cases — stress test the agent's reasoning)

Run the system on these, measure:
1. Agreement rate with expected decision
2. Rationale quality (LLM-as-judge: does the reasoning align with features?)
3. Policy citation accuracy (does the RAG agent cite correct policy clauses?)

---

## 9. IMPLEMENTATION PHASES

### Phase 1 — Foundation (Week 1-2)
- [ ] Set up LangGraph StateGraph skeleton with all nodes stubbed
- [ ] Generate 500 synthetic applications using Faker
- [ ] Build bureau JSON parser + feature extractor
- [ ] Connect AWS Bedrock (Claude Haiku for speed, Sonnet for decisions)
- [ ] Basic Streamlit UI: input form + JSON output

### Phase 2 — Agent Intelligence (Week 3-4)
- [ ] Build Bureau Analysis Agent with derived features + LLM summary
- [ ] Build Income Verification Agent (plug in your XGBoost income model)
- [ ] Build Bank Statement Agent with Pandas feature pipeline
- [ ] Implement Fraud Detection rules engine
- [ ] Set up ChromaDB with RBI policy PDFs → Policy RAG Agent

### Phase 3 — Decision Layer (Week 5)
- [ ] Implement composite scoring formula
- [ ] Build Decision Agent with Pydantic structured output
- [ ] Build Report Generation Agent (credit memo in markdown/PDF)
- [ ] Add agent trace/audit logging to state

### Phase 4 — Evaluation & Polish (Week 6)
- [ ] Build 200-case eval set, run full evaluation suite
- [ ] Tune score weights using Gini/KS on eval set
- [ ] Add latency optimization (parallel agent execution)
- [ ] Polish Streamlit demo: show live agent trace, decision confidence
- [ ] Write README + architecture diagram

---

## 10. DEMO UI DESIGN (Streamlit)

```
┌─────────────────────────────────────────────────────┐
│  🏦  CreditIQ — AI Underwriting Assistant            │
├─────────────────────────────────────────────────────┤
│  [Applicant Form]          [Live Agent Trace]        │
│  Name: ___________         ✅ Bureau Agent: Done     │
│  Loan Ask: ₹_______        ✅ Income Agent: Done     │
│  Monthly Income: ₹___      ⏳ Fraud Agent: Running   │
│  CIBIL Score: ___          ⬜ Decision Agent: Pending │
│  Employment: [SALARIED ▼]                            │
│                                                      │
│  [Upload Bank Statement PDF]                         │
│  [Run Underwriting →]                               │
├─────────────────────────────────────────────────────┤
│  DECISION: ✅ APPROVE  |  Score: 782/1000            │
│  Confidence: 89%  |  Processing: 18.3s               │
│                                                      │
│  📄 Credit Memo  |  📊 Score Breakdown  |  🔍 Trace │
└─────────────────────────────────────────────────────┘
```

---

## 11. REPO STRUCTURE

```
credit-underwriting-agent/
├── agents/
│   ├── bureau_agent.py
│   ├── income_agent.py
│   ├── bank_statement_agent.py
│   ├── fraud_agent.py
│   ├── policy_rag_agent.py
│   ├── decision_agent.py
│   └── report_agent.py
├── graph/
│   ├── state.py              # UnderwritingState TypedDict
│   └── workflow.py           # LangGraph StateGraph definition
├── data/
│   ├── synthetic/            # Generated application JSONs
│   ├── policy_docs/          # RBI PDFs, internal policy docs
│   └── eval_set/             # 200-case labeled evaluation dataset
├── models/
│   └── income_xgb.pkl        # Your existing income estimation model
├── vectorstore/
│   └── policy_chroma/        # Persisted ChromaDB
├── api/
│   └── main.py               # FastAPI endpoints
├── ui/
│   └── app.py                # Streamlit demo
├── evaluation/
│   ├── eval_runner.py        # Batch eval on 200 cases
│   └── metrics.py            # Gini, KS, precision, recall
├── notebooks/
│   └── exploration.ipynb     # EDA on synthetic data
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 12. INTERVIEW TALKING POINTS

**"Walk me through your agentic system design"**
> "I built a LangGraph-based multi-agent system with a supervisor pattern. Each agent is a specialized node — bureau, income, fraud, policy — and they run in parallel where independent. The Decision Agent aggregates all signals into a composite score using learned weights. The entire state is typed with Pydantic, making it auditable."

**"How do you prevent hallucinations in credit decisions?"**
> "Three layers: (1) structured Pydantic output schemas force the LLM into a constrained output format; (2) the Decision Agent only reasons over structured data from upstream agents, not raw text; (3) all policy citations are grounded in RAG retrieval with source tracking."

**"How did you evaluate this?"**
> "I built a 200-case synthetic eval set with known labels, measured Gini coefficient (0.52) and KS statistic (0.41) on the composite score. I also ran LLM-as-judge evals on the rationale quality — scoring reasoning coherence against expected decision logic."

**"Why LangGraph over CrewAI or AutoGen?"**
> "LangGraph gives explicit state typing and conditional edge control — critical for a compliance-sensitive domain where every decision must be auditable. AutoGen is too conversational; CrewAI abstracts away too much. LangGraph lets me see exactly what each agent received and returned."

---

## 13. STRETCH GOALS (For EF / Impressive Demo)

- **Human-in-the-loop:** Add a "Refer" flow where borderline cases are flagged for human review with pre-filled reasoning
- **Adverse Action Notices:** Auto-generate RBI-compliant rejection letters citing specific policy reasons
- **Score Migration Monitoring:** Track how composite scores shift over time as LLM versions change (model drift in agentic systems)
- **Multi-lender Comparison:** Given one application, underwrite it against 3 different lender policy sets simultaneously
- **Batch Underwriting API:** Process 100 applications/minute, store decisions in Snowflake with full audit trail

---

*Plan generated for Albert | May 2026 | Stack: LangGraph + LangChain + AWS Bedrock + ChromaDB + Snowflake*
