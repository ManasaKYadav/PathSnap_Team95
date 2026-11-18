"""Train and compare 6 candidate classifiers.

Protocol
  * 5-fold GroupKFold on the 80 training trips (groups = trip_id), so no trip is in
    both the training and validation fold.
  * Each model is then refitted on all training rows and evaluated on the 10
    student-testing trips.
  * Metrics: CV Macro F1 / Top-1 / log loss (mean and std over folds), student-test
    Top-1, Macro F1, precision, recall, log loss, route continuity, train/inference time.

Outputs (Manasa/results/): model_comparison.csv, cv_fold_scores.csv,
  confusion_matrices.csv, selected_model.json, plots/*.png

Run: python -m Manasa.train_models
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from config import MANASA_RESULTS, SEED
from Manasa.data import X, load_student_test, load_train
from Manasa.evaluate import confusion, route_continuity, row_metrics, top1_accuracy, top1_predictions
from Manasa.models import LOSS_FUNCTION, MODELS

PLOTS = MANASA_RESULTS / "plots"
N_FOLDS = 5


def cross_validate(name, factory, train, pts_per_trip, params=None):
    params = params or {}
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for fold, (tr, va) in enumerate(gkf.split(train, train.label, groups=train.trip_id)):
        model = factory(**params)
        model.fit(X(train.iloc[tr]), train.label.iloc[tr])
        val = train.iloc[va]
        proba = model.predict_proba(X(val))[:, 1]
        m = row_metrics(val.label.to_numpy(), proba)
        m["top1"] = top1_accuracy(val, proba, pts_per_trip.loc[val.trip_id.unique()].sum())
        # training-fold score on a 20k-row sample, to look for over-fitting
        tr_df = train.iloc[tr].sample(n=min(20000, len(tr)), random_state=SEED)
        tr_proba = model.predict_proba(X(tr_df))[:, 1]
        m["train_f1_macro"] = row_metrics(tr_df.label.to_numpy(), tr_proba)["f1_macro"]
        m.update(model=name, fold=fold)
        rows.append(m)
    return pd.DataFrame(rows)


def fit_and_test(name, factory, train, test, test_pts, params=None):
    params = params or {}
    model = factory(**params)
    t0 = time.perf_counter()
    model.fit(X(train), train.label)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = model.predict_proba(X(test))[:, 1]
    infer_time = time.perf_counter() - t0
    m = row_metrics(test.label.to_numpy(), proba)
    m["top1"] = top1_accuracy(test, proba, test_pts.sum())
    m["route_continuity"] = route_continuity(top1_predictions(test, proba))
    m["train_time_s"] = train_time
    m["infer_time_s"] = infer_time
    m["infer_ms_per_1k_rows"] = 1000 * infer_time / len(test) * 1000
    return model, proba, m


def plot_bars(df, col, err, title, ylabel, fname, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df.model, df[col], yerr=df[err] if err else None, capsize=4, color=color)
    for i, v in enumerate(df[col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(title=title, ylabel=ylabel)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    train, train_pts = load_train()
    test, test_pts = load_student_test()
    print(f"train rows {len(train)} ({train.trip_id.nunique()} trips), test rows {len(test)}")

    comp, folds, cms = [], [], []
    for name, (family, cfg, factory) in MODELS.items():
        t0 = time.perf_counter()
        cv = cross_validate(name, factory, train, train_pts)
        folds.append(cv)
        model, proba, m = fit_and_test(name, factory, train, test, test_pts)
        cm = confusion(test.label.to_numpy(), proba)
        cms.append(dict(model=name, tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1]))
        row = dict(model=name, family=family, configuration=cfg, loss=LOSS_FUNCTION[name],
                   cv_f1_macro=cv.f1_macro.mean(), cv_f1_macro_std=cv.f1_macro.std(),
                   cv_top1=cv.top1.mean(), cv_top1_std=cv.top1.std(),
                   cv_log_loss=cv.log_loss.mean(), cv_train_f1_macro=cv.train_f1_macro.mean())
        row.update({f"test_{k}": v for k, v in m.items()})
        comp.append(row)
        if name == "mlp":
            mlp = model[-1]
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(mlp.loss_curve_, label="training loss (log loss)", color="#4C72B0")
            ax1.set(xlabel="epoch", ylabel="training loss", title="MLP training history (train partition)")
            ax2 = ax1.twinx()
            ax2.plot(mlp.validation_scores_, label="validation accuracy (10% hold-out)", color="#DD8452")
            ax2.set_ylabel("validation accuracy")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="center right")
            fig.tight_layout()
            fig.savefig(PLOTS / "mlp_training_history.png", dpi=150)
            plt.close(fig)
        print(f"{name:20s} cvF1 {row['cv_f1_macro']:.4f} cvTop1 {row['cv_top1']:.4f} "
              f"testTop1 {m['top1']:.4f} testF1 {m['f1_macro']:.4f} ll {m['log_loss']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)")

    comp = pd.DataFrame(comp)
    comp.to_csv(MANASA_RESULTS / "model_comparison.csv", index=False)
    pd.concat(folds).to_csv(MANASA_RESULTS / "cv_fold_scores.csv", index=False)
    cms = pd.DataFrame(cms)
    cms.to_csv(MANASA_RESULTS / "confusion_matrices.csv", index=False)

    # selection: highest CV Top-1 (the task metric) -- student test is only a check
    best = comp.sort_values("cv_top1", ascending=False).iloc[0]
    sel = {"selected_model": best.model, "criterion": "highest mean 5-fold GroupKFold Top-1 accuracy",
           "cv_top1": best.cv_top1, "cv_f1_macro": best.cv_f1_macro, "test_top1": best.test_top1}
    (MANASA_RESULTS / "selected_model.json").write_text(json.dumps(sel, indent=2))
    print("selected:", sel)

    plot_bars(comp, "cv_f1_macro", "cv_f1_macro_std", "5-fold GroupKFold Macro F1 (train partition)",
              "Macro F1", "cv_f1_comparison.png", "#4C72B0")
    plot_bars(comp, "cv_top1", "cv_top1_std", "5-fold GroupKFold Top-1 accuracy (train partition)",
              "Top-1 accuracy", "cv_top1_comparison.png", "#4C72B0")
    plot_bars(comp, "test_log_loss", None, "Log loss on student-test partition (lower is better)",
              "log loss", "test_log_loss_comparison.png", "#C44E52")
    plot_bars(comp, "test_infer_ms_per_1k_rows", None, "Inference time on student-test (ms per 1000 rows)",
              "ms / 1000 rows", "inference_time.png", "#8172B2")

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    xs = np.arange(len(comp))
    ax.bar(xs - w / 2, comp.test_top1, w, label="Top-1 accuracy", color="#55A868")
    ax.bar(xs + w / 2, comp.test_f1_macro, w, label="Macro F1", color="#DD8452")
    ax.set_xticks(xs, comp.model, rotation=20)
    ax.set_ylim(0.5, 1)
    ax.set(title="Student-test performance by model", ylabel="score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "student_test_top1_f1.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (_, r) in zip(axs.ravel(), cms.iterrows()):
        mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(mat / mat.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred wrong", "pred true"])
        ax.set_yticks([0, 1], ["wrong", "true arc"])
        ax.set_title(r.model)
    fig.suptitle("Confusion matrices on student-test candidate rows (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
