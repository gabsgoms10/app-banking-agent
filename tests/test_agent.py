from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "app-banking-agent"

def test_rag_search_endpoint():
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "resolution_code": "BCB-103",
            "title": "Mecanismo Especial de Devolução (MED)",
            "category": "fraud_med",
            "content": "Bloqueio cautelar de 72h em caso de fraude."
        }
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("src.mcp_client.get_db_connection", return_value=mock_conn):
        response = client.post("/rag/search", json={"query": "MED fraude"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["match_count"] == 1
        assert data["regulations"][0]["resolution_code"] == "BCB-103"
