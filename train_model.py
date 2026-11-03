"""
Train the model that the web app serves.

Decision Tree classifier on the Breast Cancer Wisconsin dataset (bundled with scikit-learn).
To keep the app usable we keep only the N most important measurements as inputs, so the
user gets 6 sliders instead of 30.

Writes: model.joblib, model_meta.json   (run once:  python train_model.py)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42
N_FEATURES = 6
HERE = Path(__file__).parent

bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target                      # target: 0 = malignant, 1 = benign
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

# 1) rank features on the training set only, keep the top N
ranker = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
features = (pd.Series(ranker.feature_importances_, index=X.columns)
            .sort_values(ascending=False).head(N_FEATURES).index.tolist())

# 2) tune a tree on those features (5-fold CV on the training set)
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 3, 5, 10],
     "criterion": ["gini", "entropy"]},
    cv=5, scoring="accuracy").fit(X_tr[features], y_tr)
model = grid.best_estimator_

# 3) evaluate once on the untouched test set
pred = model.predict(X_te[features])
metrics = {
    "test_rows": int(len(X_te)),
    "accuracy": round(accuracy_score(y_te, pred), 3),
    "malignant_recall": round(recall_score(y_te, pred, pos_label=0), 3),
    "malignant_precision": round(precision_score(y_te, pred, pos_label=0), 3),
}

# 4) metadata the app needs for slider ranges and class comparisons
ranges = {}
for f in features:
    ranges[f] = {
        "min": float(X[f].min()), "max": float(X[f].max()),
        "median": float(X[f].median()),
        "malignant_median": float(X.loc[y == 0, f].median()),
        "benign_median": float(X.loc[y == 1, f].median()),
    }
meta = {"features": features, "ranges": ranges, "metrics": metrics,
        "params": grid.best_params_, "depth": int(model.get_depth()),
        "classes": {"0": "malignant", "1": "benign"}}

joblib.dump(model, HERE / "model.joblib")
(HERE / "model_meta.json").write_text(json.dumps(meta, indent=2))
print("Features:", features)
print("Best params:", grid.best_params_, "| depth", model.get_depth())
print("Test metrics:", metrics)
