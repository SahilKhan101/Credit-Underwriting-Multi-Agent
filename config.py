import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM — Google AI Studio
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_MODEL_FAST: str = "gemini-2.5-flash-lite"  # Analysis agents — high quota, fast
    GOOGLE_MODEL_SMART: str = "gemini-2.5-flash"      # Decision + report — better quality (20 req/day)

    # Embeddings (local HuggingFace — no API key required)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./vectorstore/policy_chroma"

    # Composite score weights (must sum to 1.0)
    BUREAU_WEIGHT: float = 0.35
    INCOME_WEIGHT: float = 0.25
    CASHFLOW_WEIGHT: float = 0.20
    FRAUD_WEIGHT: float = 0.15    # inverted — low fraud → high contribution
    POLICY_WEIGHT: float = 0.05

    # Decision thresholds (0–1000 scale)
    AUTO_APPROVE_THRESHOLD: int = 750
    APPROVE_THRESHOLD: int = 650
    REFER_THRESHOLD: int = 550
    # < 550 → AUTO_REJECT

    # Data paths
    SYNTHETIC_DATA_DIR: str = "./data/synthetic"
    POLICY_DOCS_DIR: str = "./data/policy_docs"
    EVAL_SET_DIR: str = "./data/eval_set"

    # LLM temperature
    TEMPERATURE_NARRATIVE: float = 0.1    # slight creativity for summaries
    TEMPERATURE_DECISION: float = 0.0     # deterministic decisions

    # Fraud thresholds
    CASH_WITHDRAWAL_RATIO_LIMIT: float = 0.40
    ENQUIRY_VELOCITY_LIMIT: int = 3       # enquiries in last 6m
    INCOME_DISCREPANCY_PCT_LIMIT: float = 0.20

    # Credit policy limits
    MAX_FOIR: float = 0.50                # Fixed Obligation to Income Ratio
    MIN_CIBIL: int = 550
    MAX_LOAN_TO_INCOME_RATIO: float = 18  # 18× annual monthly income
    MIN_AGE: int = 21
    MAX_AGE: int = 60


config = Config()