"""Minimal sanity check: GET /players returns a non-empty list. Run with: python -m api.tests.test_players (from repo root) or python test_players.py (from api/tests)."""
import sys
from pathlib import Path

# Ensure api dir is on path so "main" resolves to api.main
API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_players_returns_non_empty_list():
    """GET /players should return 200 and a non-empty list of player profiles."""
    response = client.get("/players")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list), "Response should be a list"
    assert len(data) > 0, "Player registry should be non-empty"
    first = data[0]
    assert "player_id" in first
    assert "label" in first
    assert "finishing" in first
    assert "passing" in first
    assert "dribbling" in first
    assert 0 <= first["finishing"] <= 1
    assert 0 <= first["passing"] <= 1
    assert 0 <= first["dribbling"] <= 1


if __name__ == "__main__":
    # Quick sanity check without pytest
    test_get_players_returns_non_empty_list()
    print("OK: GET /players returned a non-empty list.")
