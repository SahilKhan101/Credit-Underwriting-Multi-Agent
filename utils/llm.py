from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from config import config


def _check_api_key() -> None:
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add your Google AI Studio key to .env."
        )


# Conservative rate limiter: max 12 requests/min (free tier allows 15 RPM)
_rate_limiter = InMemoryRateLimiter(requests_per_second=0.2, check_every_n_seconds=0.5)


@lru_cache(maxsize=1)
def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Gemini Flash — called by each analysis agent for narrative summaries."""
    _check_api_key()
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL_FAST,
        google_api_key=config.GOOGLE_API_KEY,
        max_output_tokens=512,
        temperature=config.TEMPERATURE_NARRATIVE,
        rate_limiter=_rate_limiter,
    )


@lru_cache(maxsize=1)
def get_smart_llm() -> ChatGoogleGenerativeAI:
    """Gemini Flash — used for final credit decision and report generation."""
    _check_api_key()
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL_SMART,
        google_api_key=config.GOOGLE_API_KEY,
        max_output_tokens=2048,
        temperature=config.TEMPERATURE_DECISION,
        rate_limiter=_rate_limiter,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Local HuggingFace embeddings — no API key required."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
