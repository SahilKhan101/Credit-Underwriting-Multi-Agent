"""
Policy RAG Agent
Lazily initialises a ChromaDB vectorstore from policy documents (on first call).
Retrieves the top-4 relevant chunks for the application context and uses the LLM
to determine policy compliance, violations, and eligibility.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from graph.state import UnderwritingState
from utils.llm import get_fast_llm, get_embeddings
from config import config


@lru_cache(maxsize=1)
def _get_vectorstore():
    """Return persisted ChromaDB, building it from policy docs if needed."""
    from langchain_chroma import Chroma

    persist_dir = config.CHROMA_PERSIST_DIR
    docs_dir    = config.POLICY_DOCS_DIR

    # If already built, load and return
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())

    # Build from policy markdown files
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )

    docs: list[Document] = []
    policy_path = Path(docs_dir)
    for md_file in policy_path.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": md_file.name, "chunk": i},
            ))

    if not docs:
        # Fallback: empty store with a placeholder
        docs = [Document(page_content="No policy documents loaded.", metadata={"source": "none"})]

    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    return vs


def _build_query(profile: dict, bureau: dict) -> str:
    loan_type    = profile.get("loan_purpose", "PERSONAL")
    loan_amount  = profile.get("loan_amount", 0)
    age          = profile.get("age", 30)
    emp_type     = profile.get("employment_type", "SALARIED")
    monthly_inc  = profile.get("monthly_income", 0)
    existing_emi = bureau.get("total_existing_emi", 0)
    foir = (existing_emi + profile.get("requested_emi", 0)) / max(monthly_inc, 1)
    return (
        f"Policy eligibility for {loan_type} loan of ₹{loan_amount:,.0f} "
        f"for a {emp_type} applicant aged {age} with monthly income ₹{monthly_inc:,.0f} "
        f"and FOIR {foir:.0%}. CIBIL score {bureau.get('cibil_score', 0)}."
    )


_COMPLIANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a credit policy compliance officer at an Indian NBFC. "
        "Given the retrieved policy clauses and the loan application details, "
        "determine: (1) Is the application eligible? (2) List any policy violations. "
        "Respond in this exact format:\n"
        "ELIGIBLE: yes/no\n"
        "VIOLATIONS: <comma-separated list or 'none'>\n"
        "CITED_POLICIES: <comma-separated policy clause names>\n"
        "SUMMARY: <2-3 sentence compliance summary>",
    ),
    (
        "human",
        "Application:\n{application_context}\n\n"
        "Retrieved Policy Clauses:\n{policy_chunks}",
    ),
])


def _parse_compliance_response(text: str) -> dict:
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in text.strip().splitlines()
        if ":" in line
    }
    eligible = lines.get("ELIGIBLE", "yes").lower() == "yes"
    violations = [v.strip() for v in lines.get("VIOLATIONS", "none").split(",") if v.strip() and v.strip().lower() != "none"]
    cited = [c.strip() for c in lines.get("CITED_POLICIES", "").split(",") if c.strip()]
    summary = lines.get("SUMMARY", "Policy compliance check completed.")
    return {"eligible": eligible, "violations": violations, "cited_policies": cited, "summary": summary}


def policy_rag_node(state: UnderwritingState) -> dict:
    t0 = time.time()
    profile = state["applicant_profile"]
    bureau  = state["bureau_report"]

    try:
        vs    = _get_vectorstore()
        query = _build_query(profile, bureau)
        docs  = vs.similarity_search(query, k=4)
        chunks_text = "\n\n---\n\n".join(
            f"[{d.metadata.get('source', 'policy')}]\n{d.page_content}" for d in docs
        )

        app_context = (
            f"Loan type: {profile.get('loan_purpose', 'PERSONAL')} | "
            f"Amount: ₹{profile.get('loan_amount', 0):,.0f} | "
            f"Applicant age: {profile.get('age', 0)} | "
            f"Employment: {profile.get('employment_type', 'SALARIED')} | "
            f"CIBIL: {bureau.get('cibil_score', 0)} | "
            f"Monthly income: ₹{profile.get('monthly_income', 0):,.0f}"
        )

        chain  = _COMPLIANCE_PROMPT | get_fast_llm()
        resp   = chain.invoke({"application_context": app_context, "policy_chunks": chunks_text})
        parsed = _parse_compliance_response(resp.content)

    except Exception as exc:
        parsed = {
            "eligible": True, "violations": [],
            "cited_policies": [], "summary": f"Policy RAG unavailable: {exc}",
        }
        chunks_text = ""

    # Policy score
    n_violations = len(parsed["violations"])
    policy_score = max(0.0, 1000.0 - n_violations * 200) if parsed["eligible"] else 200.0
    policy_score = round(min(1000.0, policy_score), 2)

    flags = parsed["violations"]

    elapsed = round((time.time() - t0) * 1000)
    return {
        "policy_check": {
            "eligible": parsed["eligible"],
            "policy_violations": parsed["violations"],
            "cited_policies": parsed["cited_policies"],
            "policy_score": policy_score,
            "summary": parsed["summary"],
            "flags": flags,
        },
        "agent_trace": [f"[Policy Agent] eligible={parsed['eligible']} violations={n_violations} ({elapsed}ms)"],
    }
