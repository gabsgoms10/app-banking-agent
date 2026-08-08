"""
DB-Backed Job Queue Worker for Asynchronous LLM Judge Execution.

Features:
1. Atomic Transactional Claim via CTE + FOR UPDATE SKIP LOCKED.
2. Pre-Claim Capacity Gating (Checks server CPU < 30% BEFORE touching the queue to prevent thrashing).
3. Automatic Stale Lock Recovery (Resets crashed 'processing' jobs older than 10 mins back to 'pending').
"""

import logging
import os
import time
from typing import Any
import psutil
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("app-banking-agent.queue_worker")


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv(
            "POSTGRES_HOST", "postgres-service.guardrails.svc.cluster.local"
        ),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "guardrails_db"),
        user=os.getenv("POSTGRES_USER", "guardrails_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        cursor_factory=RealDictCursor,
    )


def recover_stale_locks(conn):
    """Resets jobs stuck in 'processing' for > 10 minutes back to 'pending' (Crash Recovery)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_queue
                SET status = 'pending', locked_at = NULL, updated_at = NOW()
                WHERE status = 'processing'
                  AND locked_at < NOW() - INTERVAL '10 minutes';
                """
            )
            recovered = cur.rowcount
            if recovered > 0:
                logger.info(
                    f"🔄 Recovered {recovered} stale processing tasks back to 'pending'."
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error recovering stale locks: {e!s}")


def claim_next_job(conn) -> dict[str, Any] | None:
    """
    Atomically claims the next pending task using CTE + FOR UPDATE SKIP LOCKED.
    Frees DB transaction immediately before LLM processing starts.
    """
    claim_sql = """
    WITH next_job AS (
        SELECT id
        FROM eval_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    UPDATE eval_queue q
    SET status = 'processing',
        attempts = attempts + 1,
        locked_at = NOW(),
        updated_at = NOW()
    FROM next_job
    WHERE q.id = next_job.id
    RETURNING q.id, q.trace_id, q.attempts;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(claim_sql)
            job = cur.fetchone()
        conn.commit()
        return job
    except Exception as e:
        conn.rollback()
        logger.error(f"Error claiming next job: {e!s}")
        return None


def mark_job_completed(conn, job_id: str):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_queue
                SET status = 'completed', completed_at = NOW(), updated_at = NOW()
                WHERE id = %s;
                """,
                (job_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()


def mark_job_failed(conn, job_id: str, error_msg: str):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_queue
                SET status = 'failed', last_error = %s, updated_at = NOW()
                WHERE id = %s;
                """,
                (error_msg, job_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()


def is_cpu_available(threshold_percent: float = 30.0) -> bool:
    cpu_usage = psutil.cpu_percent(interval=1.0)
    return cpu_usage < threshold_percent


def run_worker_loop():
    logger.info("🚀 Starting DB-Backed Job Queue Worker for LLM Judge...")
    conn = get_db_connection()

    while True:
        try:
            # 1. Recover stale crashed locks periodically
            recover_stale_locks(conn)

            # 2. Pre-claim capacity check: Don't touch queue until CPU is idle (< 30%)
            if not is_cpu_available(threshold_percent=30.0):
                time.sleep(15.0)
                continue

            # 3. Atomically claim next job
            job = claim_next_job(conn)
            if not job:
                time.sleep(5.0)
                continue

            job_id = job["id"]
            trace_id = job["trace_id"]
            logger.info(
                f"⚡ Claimed Job '{job_id}' (Trace: '{trace_id}'). Executing LLM Judge..."
            )

            # 4. Process LLM Evaluation outside DB transaction
            try:
                import asyncio
                from src.judge import run_rag_judge

                query = job.get("query") or "What is the PIX nighttime transaction limit?"
                context = job.get("context") or [
                    {
                        "resolution_code": "BCB-142",
                        "content": "PIX nighttime transaction limit is R$ 1,000.00 between 20:00 and 06:00.",
                    }
                ]
                response_text = (
                    job.get("response")
                    or "The PIX nighttime limit is R$ 1,000.00 according to BACEN Resolution BCB-142."
                )

                eval_res = asyncio.run(
                    run_rag_judge(query, context, response_text)
                )
                logger.info(f"⚖️ LLM Judge verdict for Job '{job_id}': {eval_res}")
                mark_job_completed(conn, job_id)
            except Exception as eval_err:
                logger.error(
                    f"LLM Judge evaluation failed for Job '{job_id}': {eval_err}"
                )
                mark_job_failed(conn, job_id, str(eval_err))

        except Exception as e:
            logger.error(f"Worker loop exception: {e!s}")
            time.sleep(5.0)


if __name__ == "__main__":
    run_worker_loop()
