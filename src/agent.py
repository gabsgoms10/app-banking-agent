import logging
import os
from typing import Any

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    try:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents import create_tool_calling_agent
    except Exception:
        AgentExecutor = Any
        create_tool_calling_agent = None

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.mcp_client import BANKING_TOOLS

logger = logging.getLogger("app-banking-agent.agent")

# Safe Standard OpenTelemetry / OpenInference Agent Tracing to Arize Phoenix
PHOENIX_COLLECTOR_HTTP_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
    "http://arize-phoenix-service.guardrails.svc.cluster.local:4318",
)

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{PHOENIX_COLLECTOR_HTTP_ENDPOINT}/v1/traces"
            )
        )
    )
    trace.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info(
        f"OpenTelemetry / OpenInference Agent Tracing enabled for Arize Phoenix: '{PHOENIX_COLLECTOR_HTTP_ENDPOINT}'"
    )
except Exception as e:
    logger.warning(f"Arize Phoenix Agent Tracing init deferred/skipped: {e!s}")

SYSTEM_PROMPT = """You are Enterprise X's Autonomous Banking Assistant.
You execute banking queries, account balance checks, PIX financial transfers, and regulatory searches on behalf of users.

RULES & GOVERNANCE:
1. ALWAYS verify sender balance with `get_account_balance` before attempting any transfer.
2. ALWAYS check the recipient PIX key against `check_blocked_pix_key` to avoid fraud.
3. Use `search_bacen_regulations` (Agentic RAG) whenever a user asks about transfer limits, night rules (20h-06h), or fraud procedures (MED).
4. Monetary values MUST be converted to integer cents (e.g. R$ 50,00 = 5000 cents).
5. Be polite, precise, and concise.
"""


def create_banking_agent(
    model_name: str | None = None, api_base: str | None = None
) -> Any:
    """Initializes and returns the LangChain Tool Calling Banking Agent Executor with Native Model Fallbacks & Arize Phoenix Tracing."""
    primary_model = model_name or os.getenv("LLM_MODEL", "qwen2.5:3b")
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "main_fallback")
    api_base = api_base or os.getenv(
        "OPENAI_API_BASE",
        "http://qwen-engine-service.guardrails.svc.cluster.local:11434/v1",
    )
    api_key = os.getenv("OPENAI_API_KEY", "ollama")

    logger.info(
        f"Initializing Primary LLM Engine: '{primary_model}' (Fallback: '{fallback_model}') at Endpoint: '{api_base}'"
    )

    primary_llm = ChatOpenAI(
        model_name=primary_model,
        openai_api_base=api_base,
        openai_api_key=api_key,
        temperature=0.1,
        request_timeout=30.0,
    )

    fallback_llm = ChatOpenAI(
        model_name=fallback_model,
        openai_api_base=api_base,
        openai_api_key=api_key,
        temperature=0.1,
        request_timeout=30.0,
    )

    # Resilient Fallback Chain: Switches to fallback_llm on error or timeout
    llm = primary_llm.with_fallbacks([fallback_llm])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, BANKING_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=BANKING_TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )
