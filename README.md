# PathSnap – Team 95

Map-matching and route reconstruction: noisy GPS observations are matched to the road
network with a candidate classifier followed by HMM/Viterbi decoding, served by a FastAPI
backend (Laptop 1) to a Streamlit UI (Laptop 2).

## Team and folders

| Folder | Owner | What is inside |
| --- | --- | --- |
| `Reet/` | Reet | data loading and cleaning, label alignment, candidate generation, feature engineering, EDA, Folium EDA maps |
| `Manasa/` | Manasa | 6 candidate classifiers, GroupKFold evaluation, model comparison, Optuna (Bayesian) tuning, final model |
| `Rekha/` | Rekha | road graph, HMM + Viterbi decoder, decoder tuning, classifier-only vs HMM comparison, Folium route maps |
| `Sourish/` | Sourish | FastAPI backend, Streamlit UI, deployment instructions, report draft generator |
| `config.py` | shared | paths, seed, candidate radius, thresholds |

Each folder has its own README explaining the files and the outputs.

## Pipeline

```
.track/.nodes/.arcs  -> Reet: clean, project to the trip's UTM zone, candidates within 50 m (max 8), 15 features
                     -> Manasa: classifier gives P(candidate is the true arc)
                     -> Rekha: HMM (states = candidate arcs), Viterbi -> one arc per GPS point
                     -> Sourish: API returns predicted route + server-side accuracy -> UI map
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the raw data here (not committed to Git):

```
data/raw/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
data/raw/student_test/<trip_id>/...
```

## Reproduce everything (run from the project root, in this order)

```bash
python -m Reet.candidate_study     # choose candidate radius / k (miss rate)
python -m Reet.build_dataset       # parquet datasets in Reet/output/
python -m Reet.eda                 # EDA tables, plots, Folium maps in Reet/results/
python -m Manasa.train_models      # 6 models, CV + student-test comparison
python -m Manasa.tune              # Optuna on the selected model -> Manasa/models/final_model.joblib
python -m Rekha.tune_decoder       # HMM parameters on held-out training trips -> Rekha/decoder_config.json
python -m Rekha.evaluate_hmm       # classifier-only vs HMM on student-test
python -m Rekha.maps               # Folium route maps
python -m Sourish.report.make_report   # fills report tables from the result files
```

## Run the demo (two laptops)

Laptop 1 (model + API), from the project root:

```bash
uvicorn Sourish.api.main:app --host 0.0.0.0 --port 8000
```

For the final evaluation, copy the 10 evaluator trips into a folder and point the API at it:

```bash
PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn Sourish.api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI only): `pip install -r Sourish/ui/requirements.txt`, set Laptop 1's IP address in
`Sourish/ui/config.json` (for example `http://192.168.1.20:8000`), then:

```bash
streamlit run Sourish/ui/app.py
```

Both laptops must be on the same network, and Laptop 1's firewall must allow port 8000.
See `Sourish/README.md` for details and troubleshooting.

## Rules followed

- Evaluator-testing data is never used for training, tuning or model selection.
- The prediction pipeline loads trips without the `.route` file; ground truth is read only by the
  API after prediction to compute accuracy and is never sent to the UI.
- No hardcoded predictions. Every number in the report comes from files in `*/results/`.
