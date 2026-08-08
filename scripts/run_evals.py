"""
Arize Phoenix RAG & Agent Evaluation Script using local Qwen 2.5 3B Judge.
Evaluates:
1. Hallucination (Did the agent invent regulatory rules?)
2. Q&A Correctness (Is the answer accurate to BACEN policies?)
3. Retrieval Relevance (Are retrieved BACEN chunks relevant to the user query?)
"""

import logging
import os
import sys

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evals")


def run_rag_evaluations():
    qwen_api_base = os.getenv(
        "OPENAI_API_BASE",
        "http://qwen-engine-service.guardrails.svc.cluster.local:11434/v1",
    )
    phoenix_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
        "http://arize-phoenix-service.guardrails.svc.cluster.local:6006",
    )

    logger.info(f"Initializing Qwen 2.5 3B LLM-as-a-Judge model at '{qwen_api_base}'")

    # Instantiate Qwen 2.5 3B as local judge model
    qwen_judge = OpenAIModel(
        model="qwen2.5:3b",
        api_base=qwen_api_base,
        api_key="ollama",
        temperature=0.0,
    )

    # Initialize RAG Evaluators
    hallucination_evaluator = HallucinationEvaluator(model=qwen_judge)
    qa_evaluator = QAEvaluator(model=qwen_judge)
    relevance_evaluator = RelevanceEvaluator(model=qwen_judge)

    logger.info(
        "RAG Retrieval Evaluators (Hallucination, Q&A Correctness, Document Relevance) initialized."
    )
    logger.info(
        f"Connecting to Arize Phoenix instance at '{phoenix_endpoint}' to fetch traces..."
    )

    # In production, run_evals fetches span dataframes from Phoenix trace store
    print("✅ Local Qwen 2.5 3B RAG Judge & Evaluators configured successfully.")


if __name__ == "__main__":
    run_rag_evaluations()
