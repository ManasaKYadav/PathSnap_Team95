"""Train vs test accuracy, side by side, for every model and the final tuned model.

For each model in Manasa/models.py:
  * fit on all training rows (80 trips, reliable points, every 5th GPS point)
  * TRAIN scores  = scores on those same training rows
  * CV scores     = taken from results/model_comparison.csv (unseen trips, 5-fold GroupKFold)
  * TEST scores   = scores on the 10 student-test trips
A large train-test gap means over-fitting; low train and test means under-fitting.

Outputs: results/train_vs_test.csv, plots/train_vs_test_top1.png, plots/train_vs_test_accuracy.png
Run: python -m Manasa.train_vs_test
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MANASA_RESULTS, MODEL_DIR
from Manasa.data import X, load_student_test, load_train
from Manasa.evaluate import row_metrics, top1_accuracy
from Manasa.models import MODELS
from Manasa.predict import load_model

PLOTS = MANASA_RESULTS / "plots"


def scores(model, train, train_pts, test, test_pts):
    p_tr = model.predict_proba(X(train))[:, 1]
    p_te = model.predict_proba(X(test))[:, 1]
    m_tr = row_metrics(train.label.to_numpy(), p_tr)
    m_te = row_metrics(test.label.to_numpy(), p_te)
    return dict(train_top1=top1_accuracy(train, p_tr, train_pts.sum()),
                test_top1=top1_accuracy(test, p_te, test_pts.sum()),
                train_accuracy=m_tr["accuracy"], test_accuracy=m_te["accuracy"],
                train_f1_macro=m_tr["f1_macro"], test_f1_macro=m_te["f1_macro"])


def grouped_bars(df, cols, labels, title, fname):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    n = len(cols)
    w = 0.8 / n
    xs = np.arange(len(df))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (c, lab) in enumerate(zip(cols, labels)):
        bars = ax.bar(xs + (i - (n - 1) / 2) * w, df[c], w, label=lab, color=colors[i])
        for b, v in zip(bars, df[c]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs, df.model, rotation=15)
    lo = df[cols].min().min()
    ax.set_ylim(max(0, lo - 0.08), 1.06)
    ax.set(title=title, ylabel="score")
    ax.legend(loc="upper left", ncol=n, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    train, train_pts = load_train()
    test, test_pts = load_student_test()
    try:
        cv = pd.read_csv(MANASA_RESULTS / "model_comparison.csv").set_index("model")["cv_top1"]
    except FileNotFoundError:
        cv = pd.Series(dtype=float)

    rows = []
    for name, (family, cfg, factory) in MODELS.items():
        model = factory()
        model.fit(X(train), train.label)
        r = dict(model=name, family=family, **scores(model, train, train_pts, test, test_pts))
        r["cv_top1"] = cv.get(name, np.nan)
        rows.append(r)
        print(f"{name:24s} train Top-1 {r['train_top1']:.4f}  CV Top-1 {r['cv_top1']:.4f}  "
              f"test Top-1 {r['test_top1']:.4f}  gap {r['train_top1'] - r['test_top1']:+.4f}")

    bundle = load_model()  # the final, Optuna-tuned model (already trained on the training rows)
    r = dict(model=f"{bundle['model_name']} (tuned, final)", family="final",
             **scores(bundle["model"], train, train_pts, test, test_pts))
    try:  # CV Top-1 of the tuned settings, recorded by tune.py
        r["cv_top1"] = json.loads((MODEL_DIR / "final_model_config.json").read_text())["best_cv_top1"]
    except (FileNotFoundError, KeyError):
        r["cv_top1"] = np.nan
    rows.append(r)
    print(f"{r['model']:24s} train Top-1 {r['train_top1']:.4f}  CV Top-1 {r['cv_top1']:.4f}  test Top-1 {r['test_top1']:.4f}  "
          f"gap {r['train_top1'] - r['test_top1']:+.4f}")

    df = pd.DataFrame(rows)
    df["gap_top1"] = df.train_top1 - df.test_top1
    df.to_csv(MANASA_RESULTS / "train_vs_test.csv", index=False)

    grouped_bars(df, ["train_top1", "cv_top1", "test_top1"],
                 ["Train (seen data)", "Cross-validation (unseen trips)", "Student test (10 trips)"],
                 "Top-1 accuracy: train vs cross-validation vs test", "train_vs_test_top1.png")
    grouped_bars(df, ["train_accuracy", "test_accuracy"], ["Train", "Student test"],
                 "Row-level accuracy (threshold 0.5): train vs test", "train_vs_test_accuracy.png")
    print("saved results/train_vs_test.csv and plots/train_vs_test_*.png")


if __name__ == "__main__":
    main()
