from unittest.mock import MagicMock, patch

from src.mcp_client import (
    check_blocked_pix_key,
    get_account_balance,
    search_bacen_regulations,
    transfer_pix,
)

def test_get_account_balance_success():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "Leo Vance",
        "pix_key": "leo.vance@email.com",
        "balance_cents": 250000,
        "risk_profile": "conservative"
    }
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("src.mcp_client.get_db_connection", return_value=mock_conn):
        res = get_account_balance.invoke({"pix_key": "leo.vance@email.com"})
        assert res["status"] == "success"
        assert res["name"] == "Leo Vance"
        assert res["balance_brl"] == 2500.0

def test_check_blocked_pix_key_blocked():
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "pix_key": "fraudster@pix.com",
        "reason": "PIX key flagged for malicious fraud",
        "added_at": "2026-01-01 00:00:00"
    }
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("src.mcp_client.get_db_connection", return_value=mock_conn):
        res = check_blocked_pix_key.invoke({"pix_key": "fraudster@pix.com"})
        assert res["status"] == "blocked"
        assert res["is_fraud"] is True

def test_transfer_pix_invalid_amount():
    res = transfer_pix.invoke({
        "origin_pix_key": "leo.vance@email.com",
        "destination_pix_key": "maria.silva@email.com",
        "amount_cents": -500
    })
    assert res["status"] == "error"
    assert "greater than zero" in res["message"]

def test_search_bacen_regulations_success():
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "resolution_code": "BCB-142",
            "title": "Limites de Transação Noturna PIX",
            "category": "pix_limits",
            "content": "Limite de R$ 1.000,00 entre 20h e 06h."
        }
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("src.mcp_client.get_db_connection", return_value=mock_conn):
        res = search_bacen_regulations.invoke({"query": "limite noturno"})
        assert res["status"] == "success"
        assert res["match_count"] == 1
        assert res["regulations"][0]["resolution_code"] == "BCB-142"
