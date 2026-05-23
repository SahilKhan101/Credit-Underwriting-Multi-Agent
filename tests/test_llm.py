"""
Tests for LLM connectivity and the smart/fast model configuration.
Run: .venv/bin/pytest tests/test_llm.py -v
"""
import pytest
from config import config
from utils.llm import get_fast_llm, get_smart_llm


def test_config_has_google_api_key():
    assert config.GOOGLE_API_KEY, "GOOGLE_API_KEY is empty — check .env"


def test_config_models_set():
    assert config.GOOGLE_MODEL_FAST, "GOOGLE_MODEL_FAST not set in config"
    assert config.GOOGLE_MODEL_SMART, "GOOGLE_MODEL_SMART not set in config"


def test_fast_llm_creates():
    llm = get_fast_llm()
    assert llm is not None
    assert config.GOOGLE_MODEL_FAST in llm.model


def test_smart_llm_creates():
    llm = get_smart_llm()
    assert llm is not None
    assert config.GOOGLE_MODEL_SMART in llm.model


def test_fast_llm_responds():
    llm = get_fast_llm()
    resp = llm.invoke("Reply with exactly: OK")
    assert resp.content.strip(), "Fast LLM returned empty response"


def test_smart_llm_responds():
    llm = get_smart_llm()
    resp = llm.invoke("Reply with exactly: OK")
    assert resp.content.strip(), "Smart LLM returned empty response"


def test_smart_llm_structured_output():
    """Verify with_structured_output works — this is what the decision agent uses."""
    from pydantic import BaseModel
    from typing import Literal

    class Simple(BaseModel):
        answer: Literal["yes", "no"]
        reason: str

    llm = get_smart_llm().with_structured_output(Simple)
    result = llm.invoke("Is the sky blue? Answer yes or no with a one-sentence reason.")
    assert isinstance(result, Simple)
    assert result.answer in ("yes", "no")
    assert result.reason
