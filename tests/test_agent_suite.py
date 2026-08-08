"""Unit test suite for app-banking-agent."""

from src.agent import SYSTEM_PROMPT
from src.judge import FALLBACK_JUDGE_MODELS
from src.mcp_client import BANKING_TOOLS


def test_system_prompt_rules():
    assert "RULES & GOVERNANCE" in SYSTEM_PROMPT
    assert "get_account_balance" in SYSTEM_PROMPT
    assert "check_blocked_pix_key" in SYSTEM_PROMPT


def test_judge_fallback_models():
    assert len(FALLBACK_JUDGE_MODELS) >= 3
    assert "gemini-2.5-flash" in FALLBACK_JUDGE_MODELS
    assert "gemini-1.5-flash" in FALLBACK_JUDGE_MODELS


def test_mcp_tools_defined():
    assert len(BANKING_TOOLS) > 0
