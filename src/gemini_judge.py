"""
High-Performance Asynchronous Gemini 2.5 Flash RAG Judge.
Evaluates public BACEN regulatory RAG retrieval & faithfulness with zero local CPU contention.
"""

import asyncio
import json
import logging
import os
from typing import Any
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("app-banking-agent.gemini_judge")

# Global persistent httpx client to eliminate connection overhead (Bottleneck Fix)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=45.0)
    return _http_client


class RAGEvalResult(BaseModel):
    retrieval_relevance_score: float = Field(
        ..., description="Relevance score (0.0 to 1.0) of retrieved BACEN chunks"
    )
    groundedness_score: float = Field(
        ...,
        description="Faithfulness score (0.0 to 1.0) checking for zero hallucinated rules",
    )
    answer_relevance_score: float = Field(
        ..., description="Direct responsiveness score (0.0 to 1.0) to user query"
    )
    verdict: str = Field(..., description="PASSED or FAILED")
    feedback: str = Field(..., description="Detailed technical feedback on evaluation")


async def run_gemini_rag_judge(
    user_query: str, retrieved_chunks: list[dict[str, Any]], agent_response: str
) -> dict[str, Any]:
    """
    Executes Gemini 2.5 Flash LLM-as-a-Judge for RAG Evaluation over public BACEN regulations.
    Enforces structured JSON output via Gemini 2.5 Flash Native Schema.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        logger.warning(
            "GEMINI_API_KEY not configured. Skipping Gemini 2.5 Flash evaluation."
        )
        return {"status": "skipped", "reason": "GEMINI_API_KEY missing"}

    judge_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"

    judge_system_instruction = (
        "You are an Elite Staff AI Engineer RAG Judge. Evaluate the RAG retrieval accuracy and answer faithfulness "
        "for Central Bank of Brazil (BACEN) regulations. "
        "Check if retrieved chunks match the user query, ensure zero hallucinated rules in agent_response, "
        "and return EXACTLY a JSON matching the RAGEvalResult schema."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f'USER QUERY:\n"{user_query}"\n\n'
                            f"RETRIEVED BACEN CHUNKS:\n{json.dumps(retrieved_chunks, ensure_ascii=False)}\n\n"
                            f'AGENT RESPONSE:\n"{agent_response}"'
                        )
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": judge_system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "retrieval_relevance_score": {"type": "NUMBER"},
                    "groundedness_score": {"type": "NUMBER"},
                    "answer_relevance_score": {"type": "NUMBER"},
                    "verdict": {"type": "STRING"},
                    "feedback": {"type": "STRING"},
                },
                "required": [
                    "retrieval_relevance_score",
                    "groundedness_score",
                    "answer_relevance_score",
                    "verdict",
                    "feedback",
                ],
            },
        },
    }

    client = get_http_client()
    try:
        response = await client.post(judge_url, json=payload)
        response.raise_for_status()
        data = response.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        parsed_result = json.loads(raw_text)
        logger.info(
            f"✅ Gemini 2.5 Flash RAG Judge Verdict: {parsed_result.get('verdict')}"
        )
        return parsed_result
    except Exception as e:
        logger.error(f"❌ Gemini RAG Judge Error: {e!s}")
        return {"status": "error", "message": str(e)}
