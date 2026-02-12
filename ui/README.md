# EPV Demo UI

Serve the `ui` folder with a static server. From the **repo root**:

```bash
# Python
python3 -m http.server 3000 --directory ui

# or npx (Node)
npx serve ui -p 3000
```

Open [http://localhost:3000](http://localhost:3000). Click **Fetch EPV** to call the backend (must be running on port 8000) and display EPV output.
