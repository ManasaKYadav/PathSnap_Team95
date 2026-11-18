"""Candidate classifiers compared in the experiments (6 models, 6 algorithm families).

Every model outputs P(candidate is the true arc). Models sensitive to feature
scale get a StandardScaler inside a Pipeline, so the scaler is fitted on the
training folds only.
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED

# name -> (family, short configuration description, factory(**overrides))
MODELS = {
    "logistic_regression": (
        "Linear",
        "L2, C=1.0, class_weight=balanced, scaled",
        lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(
            C=kw.get("C", 1.0), class_weight="balanced", max_iter=2000, random_state=SEED)),
    ),
    "gaussian_nb": (
        "Probabilistic",
        "var_smoothing=1e-9",
        lambda **kw: GaussianNB(var_smoothing=kw.get("var_smoothing", 1e-9)),
    ),
    "knn": (
        "Instance-based",
        "k=25, distance-weighted, Euclidean, brute-force search, scaled",
        lambda **kw: make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=kw.get("n_neighbors", 25), weights=kw.get("weights", "distance"),
            algorithm="brute", n_jobs=4)),
    ),
    "random_forest": (
        "Ensemble (bagging)",
        "200 trees, min_samples_leaf=5, class_weight=balanced_subsample",
        lambda **kw: RandomForestClassifier(
            n_estimators=kw.get("n_estimators", 200), max_depth=kw.get("max_depth", None),
            min_samples_leaf=kw.get("min_samples_leaf", 5), max_features=kw.get("max_features", "sqrt"),
            class_weight="balanced_subsample", n_jobs=4, random_state=SEED),
    ),
    "hist_gradient_boosting": (
        "Boosting",
        "max_iter=300, lr=0.05, max_leaf_nodes=31, min_samples_leaf=20",
        lambda **kw: HistGradientBoostingClassifier(
            max_iter=kw.get("max_iter", 300), learning_rate=kw.get("learning_rate", 0.05),
            max_leaf_nodes=kw.get("max_leaf_nodes", 31), min_samples_leaf=kw.get("min_samples_leaf", 20),
            l2_regularization=kw.get("l2_regularization", 0.0), random_state=SEED),
    ),
    "mlp": (
        "Neural network",
        "hidden (64, 32), ReLU, Adam lr=1e-3, early stopping, scaled",
        lambda **kw: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=kw.get("hidden_layer_sizes", (64, 32)), alpha=kw.get("alpha", 1e-4),
            learning_rate_init=kw.get("learning_rate_init", 1e-3), batch_size=512, max_iter=60,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, random_state=SEED)),
    ),
}

LOSS_FUNCTION = {
    "logistic_regression": "binary cross-entropy (log loss) + L2 penalty, class-weighted",
    "gaussian_nb": "maximum likelihood of per-class Gaussian feature distributions",
    "knn": "none (lazy learner); probability = distance-weighted vote of neighbours",
    "random_forest": "Gini impurity at each split; probabilities averaged over trees",
    "hist_gradient_boosting": "binary cross-entropy (log loss), gradient boosting on histogram-binned features",
    "mlp": "binary cross-entropy (log loss) + L2 (alpha), Adam optimiser",
}