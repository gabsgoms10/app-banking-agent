import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agent import create_banking_agent
from src.mcp_client import search_bacen_regulations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app-banking-agent.main")

app = FastAPI(
    title="Autonomous Banking Agent Service",
    description="LangChain Banking Agent service with FastMCP Tooling & Agentic RAG",
    version="1.0.0"
)

# Global Agent Executor Instance
agent_executor = None

@app.on_event("startup")
def startup_event():
    global agent_executor
    try:
        agent_executor = create_banking_agent()
        logger.info("✅ Banking Agent Executor initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {str(e)}")

class ChatRequest(BaseModel):
    message: str = Field(..., example="Qual o limite de PIX no horário noturno?")
    origin_key: str = Field("leo.vance@email.com", example="leo.vance@email.com")

class ChatResponse(BaseModel):
    status: str
    reply: str

class RAGRequest(BaseModel):
    query: str = Field(..., example="limite noturno PIX")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "app-banking-agent", "agent_ready": agent_executor is not None}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    global agent_executor
    if not agent_executor:
        agent_executor = create_banking_agent()

    logger.info(f"Received chat request: '{request.message}'")
    try:
        response = agent_executor.invoke({
            "input": f"[User Key: {request.origin_key}] {request.message}"
        })
        return ChatResponse(
            status="success",
            reply=response.get("output", "No response generated.")
        )
    except Exception as e:
        logger.error(f"Chat execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search")
def rag_search_endpoint(request: RAGRequest):
    """Direct endpoint to test Agentic Mock RAG tool."""
    return search_bacen_regulations.invoke({"query": request.query})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
