# Sourish – API, UI, deployment and report

## Files

| File | Purpose |
| --- | --- |
| `api/main.py` | FastAPI backend: `/health`, `/trips`, `/predict`; logs every request to the terminal |
| `api/test_client.py` | calls the API for every trip and prints a results table (run from any laptop) |
| `ui/app.py` | Streamlit evaluator UI: trip list, Run button, metrics, Folium map |
| `ui/config.json` | address of the API (Laptop 1's IP address and port) |
| `ui/requirements.txt` | the only packages Laptop 2 needs |
| `report/make_report.py` | builds `report/REPORT_DRAFT.md` with all tables filled from the result files |
| `results/` | `api_test_results.csv` (all 10 student-test trips through the API), `api_log.txt` (terminal log of the API), `ui_screenshot.png` |

## Two-laptop deployment

**Laptop 1: model + API**

1. Install everything and get the model files: `pip install -r requirements.txt` (project root). `Manasa/models/final_model.joblib` and `Rekha/decoder_config.json` must exist.
2. Find the IP address: `ipconfig` (Windows) or `ip addr` / `ifconfig` (Linux, Mac).
3. Start the API from the project root:

```bash
uvicorn Sourish.api.main:app --host 0.0.0.0 --port 8000
```

For the evaluator trips, point the API at the folder containing the 10 trip folders:

```bash
PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn Sourish.api.main:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell: `$env:PATHSNAP_EVAL_DIR="C:\path\to\evaluator_test"` and then the uvicorn command.

4. Allow port 8000 through the firewall. Check it works with `http://localhost:8000/docs` in a browser.

**Laptop 2: UI**

1. `pip install -r Sourish/ui/requirements.txt`
2. Put Laptop 1's address in `Sourish/ui/config.json`, for example `{"api_url": "http://192.168.1.20:8000"}`. It can also be changed in the UI sidebar.
3. `streamlit run Sourish/ui/app.py`, then open the URL it prints.

Both laptops must be on the same Wi-Fi or hotspot. Test from Laptop 2 first with `http://<laptop1-ip>:8000/trips` in a browser.

## What happens on a request

1. The UI sends `{"trip_id": ...}` to `POST /predict`.
2. The API runs Rekha's `run_pipeline`. The trip is loaded **without** the route file, then features, classifier and Viterbi run.
3. After prediction, the API loads the route on the server and computes pointwise accuracy, route continuity and route F1.
4. The response contains the predicted route coordinates, the GPS points and the metrics. It never contains the reference route.
5. The terminal on Laptop 1 logs every request (trip, accuracy, time). Show these logs to the evaluators.

## Troubleshooting

- *UI says it cannot reach the API*: wrong IP address, firewall blocking port 8000, or the laptops are on different networks.
- *404 unknown trip_id*: `PATHSNAP_EVAL_DIR` points to the wrong folder. Each trip must be a folder `<id>/` containing `<id>.track` etc.
- *Interactive map is blank*: Folium loads Leaflet and the map tiles from the internet. Without internet on Laptop 2 the interactive map stays empty; the **Static view** below it (matplotlib) always shows the GPS points and the predicted route.
