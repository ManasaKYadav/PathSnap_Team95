# Manasa – Models and Bayesian optimisation

## Files

| File | Purpose |
| --- | --- |
| `data.py` | loads Reet's parquet files (reliable points; every 5th GPS point of training trips) |
| `models.py` | the 6 candidate classifiers, their configurations and loss functions |
| `evaluate.py` | row metrics (accuracy, precision, recall, Macro F1, log loss, confusion matrix), Top-1 accuracy, route continuity |
| `train_models.py` | 5-fold GroupKFold CV + student-test evaluation for all models, plots, model selection |
| `tune.py` | Optuna (TPE) Bayesian optimisation of the selected model, tuned vs untuned, saves the final model |
| `predict.py` | `load_model()`, `predict_proba(bundle, cands)` used by Rekha's pipeline and the API |

## Run

```bash
python -m Manasa.train_models
python -m Manasa.tune          # N_TRIALS=30 by default
```

## Outputs

- `results/model_comparison.csv` one row per model: configuration, loss, CV Macro F1 / Top-1 / log loss (mean, std), training-fold F1, student-test Top-1, Macro F1, precision, recall, log loss, route continuity, training and inference time
- `results/cv_fold_scores.csv` every fold of every model
- `results/confusion_matrices.csv`, `results/selected_model.json`
- `results/optuna_trials.csv` full search record, `results/tuned_vs_untuned.csv`
- `results/plots/` CV F1, CV Top-1, student-test Top-1 + F1, log loss, inference time, confusion matrices, MLP training history, Optuna history, parameter importance
- `models/final_model.joblib` + `models/final_model_config.json`
- `results/train_log.txt`, `results/tune_log.txt` console logs of the runs

## The six models

| Model | Family | Why it was tried |
| --- | --- | --- |
| Logistic Regression | Linear | simple, well-calibrated baseline; shows how far linear boundaries go |
| Gaussian Naive Bayes | Probabilistic | very fast; tests the independence assumption (features here are correlated) |
| KNN (k=25, distance-weighted) | Instance-based | similarity in feature space; needs scaling; slow inference |
| Random Forest | Ensemble (bagging) | non-linear, robust to scale, little tuning |
| LightGBM | Boosting | sequential error correction, fast, usually strongest on tabular data |
| MLP (64, 32) | Neural network | non-linear representation; has an epoch-wise training history |

## Key decisions (be ready to explain)

1. **The task as binary classification.** Arc ids differ per trip, so each (GPS point, candidate arc) row is classified as "true arc" or not. The arc with the highest probability is the point's prediction.
2. **GroupKFold by trip.** Points from the same trip are strongly correlated; mixing them across folds would leak information and inflate scores.
3. **Primary metric: Top-1 accuracy** (the share of GPS points whose highest-scoring candidate is the true arc). It measures the real task. Log loss is the second check because the HMM uses the probabilities. Macro F1 at threshold 0.5 is reported as required.
4. **Selection** uses CV Top-1, not student-test scores, so the student-test partition stays a fair check.
5. **Bayesian optimisation.** Optuna's TPE sampler builds probability models of good vs bad parameter regions from past trials and samples where improvement is likely. The objective is the same CV Top-1; the student-test set is not touched during the search.
6. **Note on KNN training score.** With distance weighting, each training point is its own nearest neighbour (distance 0), so its training-fold score is close to 1 by construction. This is not evidence of over-fitting by itself.
