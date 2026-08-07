import os
import logging
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.tools import tool

logger = logging.getLogger("app-banking-agent.mcp_client")

# FastMCP Server Base URL
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-banking-service.guardrails.svc.cluster.local:8001")

def get_db_connection():
    """Fallback connection to PostgreSQL if direct DB execution is requested."""
    db_host = os.getenv("POSTGRES_HOST", "postgres-service.guardrails.svc.cluster.local")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "guardrails_db")
    db_user = os.getenv("POSTGRES_USER", "guardrails_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "")

    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        cursor_factory=RealDictCursor
    )

@tool
def get_account_balance(pix_key: str) -> dict:
    """
    Retrieves account details and current balance (in BRL and cents) for a given PIX key or character name.
    
    Args:
        pix_key: The PIX key (e.g., 'leo.vance@email.com', 'maria.silva@email.com') or account name.
    """
    logger.info(f"[Agent Tool] Querying account balance for: '{pix_key}'")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, pix_key, balance_cents, risk_profile FROM characters WHERE pix_key = %s OR name ILIKE %s;",
                (pix_key, f"%{pix_key}%")
            )
            account = cur.fetchone()
        conn.close()

        if account:
            return {
                "status": "success",
                "account_id": account["id"],
                "name": account["name"],
                "pix_key": account["pix_key"],
                "balance_cents": account["balance_cents"],
                "balance_brl": account["balance_cents"] / 100.0,
                "risk_profile": account["risk_profile"]
            }
        return {"status": "error", "message": f"Account not found for PIX key: {pix_key}"}
    except Exception as e:
        logger.error(f"[Agent Tool Error] get_account_balance failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@tool
def check_blocked_pix_key(pix_key: str) -> dict:
    """
    Checks if a destination PIX key is flagged in the BACEN fraud registry (blocked_pix_keys).
    
    Args:
        pix_key: The PIX key to verify against the fraud registry (e.g. 'fraudster@pix.com').
    """
    logger.info(f"[Agent Tool] Checking fraud registry for key: '{pix_key}'")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pix_key, reason, added_at FROM blocked_pix_keys WHERE pix_key = %s;",
                (pix_key,)
            )
            blocked = cur.fetchone()
        conn.close()

        if blocked:
            return {
                "status": "blocked",
                "is_fraud": True,
                "pix_key": blocked["pix_key"],
                "reason": blocked["reason"],
                "added_at": str(blocked["added_at"])
            }
        return {
            "status": "clean",
            "is_fraud": False,
            "pix_key": pix_key,
            "message": "PIX key is clear for financial transfer"
        }
    except Exception as e:
        logger.error(f"[Agent Tool Error] check_blocked_pix_key failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@tool
def transfer_pix(origin_pix_key: str, destination_pix_key: str, amount_cents: int) -> dict:
    """
    Executes an instant PIX financial transfer between accounts.
    Strictly uses integer cents for monetary integrity (e.g. 50000 = R$ 500.00).
    
    Args:
        origin_pix_key: Sender's PIX key or character name.
        destination_pix_key: Recipient's PIX key.
        amount_cents: Transfer amount in integer cents (must be > 0).
    """
    logger.info(f"[Agent Tool] Initiating transfer: {amount_cents} cents from '{origin_pix_key}' to '{destination_pix_key}'")
    if amount_cents <= 0:
        return {"status": "error", "message": "Transfer amount_cents must be greater than zero"}

    try:
        conn = get_db_connection()
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, balance_cents FROM characters WHERE pix_key = %s OR name ILIKE %s FOR UPDATE;", (origin_pix_key, f"%{origin_pix_key}%"))
            origin = cur.fetchone()

            if not origin:
                conn.rollback()
                return {"status": "error", "message": f"Sender account '{origin_pix_key}' not found"}

            if origin["balance_cents"] < amount_cents:
                conn.rollback()
                return {"status": "error", "message": f"Insufficient funds. Balance: {origin['balance_cents']} cents, Requested: {amount_cents} cents"}

            cur.execute("SELECT reason FROM blocked_pix_keys WHERE pix_key = %s;", (destination_pix_key,))
            blocked = cur.fetchone()
            if blocked:
                cur.execute(
                """
                INSERT INTO transactions (
                    origin_character_id, destination_key, amount_cents, status, decisive_rail, reason
                ) VALUES (%s, %s, %s, 'blocked', 'BACEN_FRAUD_LIST', %s);
                """,
                (origin["id"], destination_pix_key, amount_cents, f"Blocked key: {blocked['reason']}")
            )
            conn.commit()
            conn.close()
            return {
                "status": "blocked",
                "reason": (
                    f"Destination key '{destination_pix_key}' is blocked by BACEN fraud registry: "
                    f"{blocked['reason']}"
                )
            }

        cur.execute(
            "UPDATE characters SET balance_cents = balance_cents - %s WHERE id = %s;",
            (amount_cents, origin["id"])
        )
        cur.execute(
            "UPDATE characters SET balance_cents = balance_cents + %s WHERE pix_key = %s;",
            (amount_cents, destination_pix_key)
        )
        cur.execute(
            """
            INSERT INTO transactions (
                origin_character_id, destination_key, amount_cents, status, decisive_rail, reason
            ) VALUES (%s, %s, %s, 'approved', 'EXECUTION_RAIL', 'Transaction executed successfully');
            """,
            (origin["id"], destination_pix_key, amount_cents)
        )

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": f"Successfully transferred {amount_cents / 100.0:.2f} BRL to {destination_pix_key}",
            "amount_cents": amount_cents,
            "new_origin_balance_cents": origin["balance_cents"] - amount_cents
        }

    except Exception as e:
        logger.error(f"[Agent Tool Error] transfer_pix aborted: {str(e)}")
        return {"status": "error", "message": str(e)}

@tool
def search_bacen_regulations(query: str) -> dict:
    """
    Performs an Agentic RAG search over Central Bank (BACEN) regulations,
    PIX night transfer limits, and MED fraud policies.

    Args:
        query: Search prompt or question regarding financial rules.
    """
    logger.info(f"[Agent Tool - RAG] Searching BACEN regulations for query: '{query}'")
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT resolution_code, title, category, content
                FROM bacen_regulations
                WHERE title ILIKE %s OR content ILIKE %s OR %s = ANY(keywords)
                ORDER BY created_at DESC;
                """,
                (f"%{query}%", f"%{query}%", query.lower())
            )
            results = cur.fetchall()
        conn.close()

        if results:
            formatted_results = [
                {
                    "resolution_code": r["resolution_code"],
                    "title": r["title"],
                    "category": r["category"],
                    "content": r["content"]
                }
                for r in results
            ]
            return {
                "status": "success",
                "query": query,
                "match_count": len(formatted_results),
                "regulations": formatted_results
            }

        return {
            "status": "success",
            "query": query,
            "match_count": 0,
            "regulations": [],
            "message": f"No specific BACEN regulation matched query '{query}'."
        }
    except Exception as e:
        logger.error(f"[Agent Tool Error] search_bacen_regulations failed: {str(e)}")
        return {"status": "error", "message": str(e)}

# Export toolset for LangChain Agentic Loop
BANKING_TOOLS = [
    get_account_balance,
    check_blocked_pix_key,
    transfer_pix,
    search_bacen_regulations
]
