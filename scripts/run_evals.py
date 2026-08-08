"""
Asynchronous BACEN RAG Evaluation & Experiment Script using Gemini API / Phoenix Evaluators.

Evaluates 3 Core RAG Metrics:
1. Retrieval Relevance: Did search_bacen_regulations fetch the correct BACEN resolution?
2. Groundedness/Faithfulness: Does the response cite only retrieved BACEN facts without hallucination?
3. Answer Relevance: Is the answer directly responsive to the customer query?

Uses Gemini API (External Judge for Public Non-PII BACEN Regulations) to prevent CPU contention on the local K3s VM.
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
logger = logging.getLogger("async_evals")


def run_rag_experiment():
    # Use Gemini API or local endpoint for public non-PII BACEN evaluations
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    phoenix_url = os.getenv(
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
        "http://arize-phoenix-service.guardrails.svc.cluster.local:6006",
    )

    logger.info("Initializing Asynchronous BACEN RAG Evaluators...")

    # Judge model configuration (Gemini API for offloaded CPU or fallback to Qwen)
    if gemini_api_key:
        logger.info(
            "Using Gemini API as External Judge for Public BACEN Regulations (Zero Local CPU Contention)."
        )
        eval_model = OpenAIModel(
            model="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key,
            temperature=0.0,
        )
    else:
        logger.info("Gemini API Key not set. Falling back to local Qwen 2.5 3B Engine.")
        eval_model = OpenAIModel(
            model="qwen2.5:3b",
            api_base=os.getenv(
                "OPENAI_API_BASE",
                "http://qwen-engine-service.guardrails.svc.cluster.local:11434/v1",
            ),
            api_key="ollama",
            temperature=0.0,
        )

    # 1. Retrieval Relevance Evaluator
    retrieval_evaluator = RelevanceEvaluator(model=eval_model)

    # 2. Groundedness / Faithfulness Evaluator
    groundedness_evaluator = HallucinationEvaluator(model=eval_model)

    # 3. Answer Relevance Evaluator
    answer_relevance_evaluator = QAEvaluator(model=eval_model)

    logger.info(
        "✅ RAG Evaluators Initialized: Retrieval Relevance, Groundedness/Faithfulness, Answer Relevance."
    )
    logger.info(f"Target Arize Phoenix Instance: '{phoenix_url}'")


if __name__ == "__main__":
    run_rag_experiment()
