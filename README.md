# 🤖 app-banking-agent

> **Autonomous Banking Agent Service**: LangChain ReAct / OpenAI Tools Agent microservice connected to the **FastMCP Banking Tools Server** and **PostgreSQL pgvector Agentic RAG**.

---

## 🎯 Architectural Thesis & Features

1. **ReAct Agentic Loop (LangChain)**:
   - Evaluates user banking intent and executes multi-step reasoning.
   - Automatically invokes tools: `get_account_balance`, `check_blocked_pix_key`, `transfer_pix`, and `search_bacen_regulations`.

2. **Agentic Mock RAG (`pgvector`)**:
   - Queries `bacen_regulations` in PostgreSQL to fetch Central Bank policy resolutions (e.g. Resolução BCB nº 142 on night-time R$ 1.000 PIX limits and MED fraud rules).

3. **FastAPI Exposure**:
   - Provides `/chat` and `/rag/search` endpoints for NeMo Guardrails and the Web Playground.
