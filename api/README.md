# EPV Demo API

From the **repo root**:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- **GET** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — returns `{ "status": "ok" }`
- **POST** [http://127.0.0.1:8000/api/epv](http://127.0.0.1:8000/api/epv) — returns placeholder EPV values (no body required)

CORS is enabled for `http://localhost:3000` so the frontend can call the API.
