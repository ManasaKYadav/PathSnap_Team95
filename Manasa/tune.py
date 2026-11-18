"""Bayesian hyperparameter optimisation (Optuna, TPE sampler) of the selected model.

  * search space: defined per model family below
  * objective: mean 5-fold GroupKFold Top-1 accuracy on the 80 training trips
  * the student-test partition is NOT used during the search; it is only used to
    compare the untuned and tuned models afterwards
  * the tuned model is refitted on all training rows and saved as
    Manasa/models/final_model.joblib (+ final_model_config.json)

Outputs (Manasa/results/): optuna_trials.csv, tuned_vs_untuned.csv,
  plots/optuna_history.png, plots/optuna_param_importance.png

Run: python -m Manasa.tune            (N_TRIALS env var, default 30)
"""
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from config import MANASA_RESULTS, MODEL_DIR, SEED
from Manasa.data import X, load_student_test, load_train
from Manasa.evaluate import top1_accuracy
from Manasa.models import MODELS
from Manasa.train_models import cross_validate, fit_and_test
from Reet.features import FEATURE_COLUMNS

N_TRIALS = int(os.environ.get("N_TRIALS", 30))
PLOTS = MANASA_RESULTS / "plots"

# MLP layer options as strings (Optuna categorical values must be simple types)
MLP_LAYERS = {"32": (32,), "64-32": (64, 32), "128-64": (128, 64), "128-64-32": (128, 64, 32)}


def search_space(name, trial):
    if name == "hist_gradient_boosting":
        return dict(
            max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        )
    if name == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30, log=True),
            max_features=trial.suggest_float("max_features", 0.2, 1.0),
        )
    if name == "mlp":
        layers = trial.suggest_categorical("hidden_layer_sizes", list(MLP_LAYERS))
        return dict(
            hidden_layer_sizes=MLP_LAYERS[layers],
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    if name == "knn":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 5, 101, step=2),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
    if name == "logistic_regression":
        return dict(C=trial.suggest_float("C", 1e-3, 100, log=True))
    if name == "gaussian_nb":
        return dict(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True))
    raise ValueError(f"no search space defined for model '{name}'")


def cv_top1(factory, params, train, pts_per_trip):
    scores = []
    for tr, va in GroupKFold(n_splits=5).split(train, train.label, groups=train.trip_id):
        model = factory(**params)
        model.fit(X(train.iloc[tr]), train.label.iloc[tr])
        val = train.iloc[va]
        p = model.predict_proba(X(val))[:, 1]
        scores.append(top1_accuracy(val, p, pts_per_trip.loc[val.trip_id.unique()].sum()))
    return float(np.mean(scores))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sel = json.loads((MANASA_RESULTS / "selected_model.json").read_text())
    name = sel["selected_model"]
    if name not in MODELS:
        raise SystemExit(f"selected model '{name}' is not in Manasa/models.py - rerun python -m Manasa.train_models")
    family, cfg, factory = MODELS[name]
    train, train_pts = load_train()
    test, test_pts = load_student_test()
    print(f"tuning {name} with {N_TRIALS} trials")

    def objective(trial):
        return cv_top1(factory, search_space(name, trial), train, train_pts)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name=f"pathsnap_{name}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d} top1={t.value:.4f} best={s.best_value:.4f}")])
    search_time = time.perf_counter() - t0
    trials = study.trials_dataframe()
    trials.to_csv(MANASA_RESULTS / "optuna_trials.csv", index=False)

    best_params = search_space(name, optuna.trial.FixedTrial(study.best_params))
    print("best params:", best_params)

    # untuned vs tuned, same protocol and partitions
    rows = []
    final_model = None
    for label, params in [("untuned", {}), ("tuned", best_params)]:
        cv = cross_validate(name, factory, train, train_pts, params)
        model, proba, m = fit_and_test(name, factory, train, test, test_pts, params)
        rows.append(dict(version=label, cv_f1_macro=cv.f1_macro.mean(), cv_top1=cv.top1.mean(),
                         cv_log_loss=cv.log_loss.mean(), cv_train_f1_macro=cv.train_f1_macro.mean(),
                         test_top1=m["top1"], test_f1_macro=m["f1_macro"], test_log_loss=m["log_loss"],
                         test_route_continuity=m["route_continuity"], train_time_s=m["train_time_s"],
                         infer_ms_per_1k_rows=m["infer_ms_per_1k_rows"]))
        if label == "tuned":
            final_model = model
    comp = pd.DataFrame(rows)
    comp.to_csv(MANASA_RESULTS / "tuned_vs_untuned.csv", index=False)
    print(comp.T)

    bundle = dict(model=final_model, model_name=name, params=best_params, feature_columns=FEATURE_COLUMNS)
    joblib.dump(bundle, MODEL_DIR / "final_model.joblib", compress=3)
    cfg_out = dict(model_name=name, family=family,
                   params={k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
                   feature_columns=FEATURE_COLUMNS, n_trials=N_TRIALS, sampler="TPE (seed 42)",
                   objective="mean 5-fold GroupKFold Top-1 accuracy (train partition)",
                   best_cv_top1=float(study.best_value), search_time_s=search_time,
                   trained_on="80 training trips, reliable points, every 5th GPS point")
    (MODEL_DIR / "final_model_config.json").write_text(json.dumps(cfg_out, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = trials.value.to_numpy()
    ax.scatter(trials.number, vals, s=14, label="trial value", color="#4C72B0")
    ax.plot(trials.number, np.maximum.accumulate(vals), color="#C44E52", label="best so far")
    ax.set(title=f"Optuna optimisation history ({name}, TPE)", xlabel="trial",
           ylabel="CV Top-1 accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_history.png", dpi=150)
    plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#55A868")
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        fig.tight_layout()
        fig.savefig(PLOTS / "optuna_param_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # importance needs >1 distinct trial
        print("importance plot skipped:", e)


if __name__ == "__main__":
    main()