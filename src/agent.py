import logging
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.mcp_client import BANKING_TOOLS

logger = logging.getLogger("app-banking-agent.agent")

# Safe MLflow Agent Tracing Import
try:
    import mlflow
    import mlflow.langchain

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow module not found. Agent Tracing will be disabled.")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.guardrails.svc.cluster.local:5000",
)

if HAS_MLFLOW:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("banking-agent-tracing")
        mlflow.langchain.autolog(log_traces=True)
        logger.info(f"MLflow Agent Tracing enabled at URI: '{MLFLOW_TRACKING_URI}'")
    except Exception as e:
        logger.warning(f"MLflow Agent Tracing init deferred: {e!s}")

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
) -> AgentExecutor:
    """Initializes and returns the LangChain Tool Calling Banking Agent Executor with MLflow Tracing."""
    model_name = model_name or os.getenv("LLM_MODEL", "qwen2.5:3b")
    api_base = api_base or os.getenv(
        "OPENAI_API_BASE",
        "http://qwen-engine-service.guardrails.svc.cluster.local:11434/v1",
    )
    api_key = os.getenv("OPENAI_API_KEY", "ollama")

    logger.info(
        f"Initializing LangChain Banking Agent with Model: '{model_name}' at Endpoint: '{api_base}'"
    )

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_base=api_base,
        openai_api_key=api_key,
        temperature=0.1,
    )

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
