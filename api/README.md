# EPV Demo API

From the **repo root**:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- **GET** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — returns `{ "status": "ok" }`
- **GET** [http://127.0.0.1:8000/players](http://127.0.0.1:8000/players) — returns player profile registry (normalized finishing/passing/dribbling from skill CSVs)
- **POST** [http://127.0.0.1:8000/api/epv](http://127.0.0.1:8000/api/epv) — returns placeholder EPV values (no body required)

CORS is enabled for `http://localhost:3000` so the frontend can call the API.

**Tests (from repo root with venv active):** `pytest api/tests/ -v` or `cd api && python tests/test_players.py` for a quick sanity check that GET /players returns a non-empty list.
