"""
CPU-Aware Asynchronous RAG Evaluation & Experiment Script using local Qwen 2.5 3B Judge.

Executes 3 Core RAG Metrics (Retrieval Relevance, Groundedness/Faithfulness, Answer Relevance)
ONLY when server CPU utilization drops below threshold (< 30%), preventing resource contention
with live agent & guardrail requests on the local K3s VM.
"""

import logging
import os
import time
import psutil

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cpu_aware_evals")


def wait_for_low_cpu_idle(
    threshold_percent: float = 30.0, check_interval_sec: int = 10
):
    """Monitors CPU usage and blocks until server CPU drops below threshold."""
    logger.info(
        f"Checking server CPU utilization (Target Threshold: < {threshold_percent}%)..."
    )
    while True:
        cpu_usage = psutil.cpu_percent(interval=2.0)
        if cpu_usage < threshold_percent:
            logger.info(
                f"✅ Low CPU load detected ({cpu_usage:.1f}%). Safe to launch local LLM Judge!"
            )
            break
        logger.info(
            f"⏳ Current CPU usage is {cpu_usage:.1f}% (exceeds {threshold_percent}% limit). "
            f"Waiting {check_interval_sec}s for idle state..."
        )
        time.sleep(check_interval_sec)


def run_rag_experiment():
    phoenix_url = os.getenv(
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
        "http://arize-phoenix-service.guardrails.svc.cluster.local:6006",
    )
    qwen_api_base = os.getenv(
        "OPENAI_API_BASE",
        "http://qwen-engine-service.guardrails.svc.cluster.local:11434/v1",
    )

    # 1. Gate execution: Wait until server is idle (< 30% CPU)
    wait_for_low_cpu_idle(threshold_percent=30.0)

    logger.info(
        f"Initializing local Qwen 2.5 3B LLM-as-a-Judge model at '{qwen_api_base}'..."
    )

    # Local Qwen 2.5 3B Judge Instance
    qwen_judge = OpenAIModel(
        model="qwen2.5:3b",
        api_base=qwen_api_base,
        api_key="ollama",
        temperature=0.0,
    )

    # 2. Initialize RAG Evaluators
    retrieval_evaluator = RelevanceEvaluator(model=qwen_judge)
    groundedness_evaluator = HallucinationEvaluator(model=qwen_judge)
    answer_relevance_evaluator = QAEvaluator(model=qwen_judge)

    logger.info(
        "✅ Local RAG Evaluators Initialized: Retrieval Relevance, Groundedness/Faithfulness, Answer Relevance."
    )
    logger.info(f"Target Arize Phoenix Instance: '{phoenix_url}'")


if __name__ == "__main__":
    run_rag_experiment()
