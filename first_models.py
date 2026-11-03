"""
First machine-learning models with scikit-learn.

  A. Decision Tree CLASSIFIER  - Breast Cancer Wisconsin (is the tumour malignant?)
  B. Linear Regression         - Diabetes (predict disease progression after one year)
     (compared with a Decision Tree regressor and a predict-the-mean baseline)

Both datasets ship with scikit-learn, so no download is needed.
Usage:  python first_models.py [output_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

SEED = 42
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
OUT.mkdir(parents=True, exist_ok=True)

# ---- style (validated blue/orange categorical slots, light surface) ----------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def header(fig, title, subtitle, left=0.06):
    fig.text(left, 0.97, title, fontsize=15, fontweight="bold", va="top")
    fig.text(left, 0.925, subtitle, fontsize=10.5, va="top", color=INK2)


def legend_row(fig, items, y=0.885, left=0.06):
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, y), ncol=len(items),
               fontsize=10.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0,
               columnspacing=1.6)


results = {}

# =============================================================================
# A. DECISION TREE CLASSIFIER
# =============================================================================
bc = load_breast_cancer(as_frame=True)
X, y = bc.data, bc.target                      # target: 0 = malignant, 1 = benign
class_names = list(bc.target_names)            # ['malignant', 'benign']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"Classification: {len(X_tr)} train / {len(X_te)} test rows; "
      f"{(y == 0).mean() * 100:.1f}% malignant overall")

# baseline: always predict the majority class
base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
base_acc = accuracy_score(y_te, base.predict(X_te))

# 1) unconstrained tree -> overfits
deep = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
deep_train, deep_test = deep.score(X_tr, y_tr), deep.score(X_te, y_te)

# 2) tuned tree: pick depth / leaf size with 5-fold CV on the TRAIN set only
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 2, 5, 10],
     "criterion": ["gini", "entropy"]},
    cv=5, scoring="accuracy").fit(X_tr, y_tr)
tree = grid.best_estimator_
pred = tree.predict(X_te)
proba_malignant = tree.predict_proba(X_te)[:, list(tree.classes_).index(0)]
cm = confusion_matrix(y_te, pred, labels=[0, 1])   # rows actual, cols predicted

results["classifier"] = {
    "baseline_majority_accuracy": round(base_acc, 3),
    "unconstrained_tree": {"train_acc": round(deep_train, 3), "test_acc": round(deep_test, 3),
                           "depth": int(deep.get_depth())},
    "best_params": grid.best_params_,
    "cv_accuracy_train": round(grid.best_score_, 3),
    "test": {
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "malignant_precision": round(precision_score(y_te, pred, pos_label=0), 3),
        "malignant_recall": round(recall_score(y_te, pred, pos_label=0), 3),
        "malignant_f1": round(f1_score(y_te, pred, pos_label=0), 3),
        "roc_auc": round(roc_auc_score((y_te == 0).astype(int), proba_malignant), 3),
    },
    "confusion_matrix [[TP_mal, FN],[FP, TN_ben]]": cm.tolist(),
}
imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
results["classifier"]["top_features"] = {k: round(v, 3) for k, v in imp.head(5).items()}

# depth curve (train vs 5-fold CV, train set only)
depths = list(range(1, 11))
tr_curve, cv_curve = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, criterion=grid.best_params_["criterion"],
                               random_state=SEED)
    tr_curve.append(m.fit(X_tr, y_tr).score(X_tr, y_tr))
    cv_curve.append(cross_val_score(m, X_tr, y_tr, cv=5).mean())

# =============================================================================
# B. LINEAR REGRESSION (+ tree regressor + baseline)
# =============================================================================
db = load_diabetes(as_frame=True)
Xr, yr = db.data, db.target                    # features are already mean-centred and scaled
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
print(f"Regression: {len(Xr_tr)} train / {len(Xr_te)} test rows")


def reg_metrics(model):
    p = model.predict(Xr_te)
    return {"r2": round(r2_score(yr_te, p), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(yr_te, p))), 1),
            "mae": round(mean_absolute_error(yr_te, p), 1)}


mean_base = DummyRegressor(strategy="mean").fit(Xr_tr, yr_tr)
lin = LinearRegression().fit(Xr_tr, yr_tr)
cv_r2 = cross_val_score(LinearRegression(), Xr_tr, yr_tr, cv=5, scoring="r2")
tree_grid = GridSearchCV(DecisionTreeRegressor(random_state=SEED),
                         {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [5, 10, 20, 30]},
                         cv=5, scoring="r2").fit(Xr_tr, yr_tr)
results["regression"] = {
    "baseline_mean": reg_metrics(mean_base),
    "linear_regression": {**reg_metrics(lin), "cv_r2_mean": round(cv_r2.mean(), 3),
                          "cv_r2_std": round(cv_r2.std(), 3),
                          "train_r2": round(lin.score(Xr_tr, yr_tr), 3)},
    "decision_tree_regressor": {**reg_metrics(tree_grid.best_estimator_),
                                "best_params": tree_grid.best_params_},
}
coef = pd.Series(lin.coef_, index=Xr.columns).sort_values()
results["regression"]["top_coefficients"] = {k: round(v, 1) for k, v in
                                             coef.reindex(coef.abs().sort_values(ascending=False).index).head(4).items()}

results["classifier"]["depth_curve_cv_acc"] = {d: round(c, 3) for d, c in zip(depths, cv_curve)}
print(json.dumps(results, indent=2))
(OUT / "results.json").write_text(json.dumps(results, indent=2))

# =============================================================================
# CHARTS
# =============================================================================
# 1. depth curve
fig, ax = plt.subplots(figsize=(9, 5.4))
fig.subplots_adjust(top=0.78, bottom=0.13, left=0.09, right=0.95)
ax.plot(depths, np.array(tr_curve) * 100, color=ORANGE, lw=2.5)
ax.plot(depths, np.array(cv_curve) * 100, color=BLUE, lw=2.5)
best_d = grid.best_params_["max_depth"]
ax.axvline(best_d, color=BASE, lw=1)
ax.text(best_d + .1, 90.5, f"tuned depth = {best_d}", color=INK2, fontsize=10)
ax.set_ylim(88, 101)
ax.set_xticks(depths)
ax.set_xlabel("max_depth (tree complexity)")
ax.set_ylabel("Accuracy (%)")
ax.grid(axis="x", visible=False)
gap = (tr_curve[-1] - cv_curve[-1]) * 100
plateau = next(d for d, c in zip(depths, cv_curve) if c >= max(cv_curve) - 0.01)
header(fig, f"Accuracy on held-out data levels off by depth {plateau}",
       f"Training accuracy reaches {tr_curve[-1] * 100:.0f}% at depth 10, but cross-validated accuracy is "
       f"{cv_curve[-1] * 100:.0f}% ({gap:.0f}-point gap)")
results["classifier"]["cv_plateau_depth"] = plateau
legend_row(fig, [("Training accuracy", ORANGE), ("Cross-validated accuracy (5-fold)", BLUE)])
fig.savefig(OUT / "01_tree_depth_curve.png", dpi=160)
plt.close(fig)

# 2. confusion matrix + feature importances
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.3]})
fig.subplots_adjust(top=0.76, bottom=0.12, left=0.08, right=0.97, wspace=0.45)
a1.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max() * 1.05)
for i in range(2):
    for j in range(2):
        a1.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=22, fontweight="bold",
                color="#ffffff" if cm[i, j] > cm.max() * .45 else INK)
a1.set_xticks([0, 1], [f"Predicted\n{c}" for c in class_names])
a1.set_yticks([0, 1], [f"Actual\n{c}" for c in class_names])
a1.grid(False)
a1.tick_params(length=0, labelsize=10, colors=INK2)
for s in a1.spines.values():
    s.set_visible(False)
a1.set_title("Confusion matrix (test set)", loc="left", fontsize=11.5, pad=10)
top = imp.head(8).sort_values()
a2.barh(top.index, top.values, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
for i, v in enumerate(top.values):
    a2.text(v + .005, i, f"{v:.2f}", va="center", fontsize=9.5, color=INK2)
a2.grid(axis="y", visible=False)
a2.grid(axis="x", visible=True)
a2.tick_params(axis="y", length=0, colors=INK2)
a2.set_xlabel("Feature importance")
a2.set_title("What the tree relies on", loc="left", fontsize=11.5, pad=10)
t = results["classifier"]["test"]
header(fig, f"The tuned tree catches {t['malignant_recall'] * 100:.0f}% of malignant tumours on unseen data",
       f"Test accuracy {t['accuracy'] * 100:.1f}% vs {base_acc * 100:.1f}% for always guessing benign; "
       f"malignant precision {t['malignant_precision'] * 100:.1f}%")
fig.savefig(OUT / "02_tree_confusion_importances.png", dpi=160)
plt.close(fig)

# 3. tree diagram (top three levels)
fig, ax = plt.subplots(figsize=(15, 7.2))
fig.subplots_adjust(top=0.86, bottom=0.02, left=0.01, right=0.99)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
arts = plot_tree(tree, max_depth=3, feature_names=list(X.columns), class_names=class_names,
                 filled=True, rounded=True, fontsize=9, ax=ax, proportion=False, impurity=False)
for a in arts:
    txt = a.get_text()
    a.set_bbox(dict(boxstyle="round,pad=0.35", ec=BASE,
                    fc="#fbd9ce" if "class = malignant" in txt else "#cde2fb"))
    a.set_color(INK)
header(fig, "How the tuned tree decides (top three levels)",
       f"Each box asks one question about a measurement; orange boxes lean malignant, blue lean benign "
       f"(full tree has depth {tree.get_depth()})", left=0.02)
fig.savefig(OUT / "03_tree_diagram.png", dpi=150)
plt.close(fig)

# 4. linear regression: predicted vs actual + coefficients
pred_r = lin.predict(Xr_te)
lr = results["regression"]["linear_regression"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
fig.subplots_adjust(top=0.76, bottom=0.2, left=0.07, right=0.97, wspace=0.35)
lo, hi = min(yr_te.min(), pred_r.min()) - 10, max(yr_te.max(), pred_r.max()) + 10
a1.plot([lo, hi], [lo, hi], color=BASE, lw=1.5)
a1.scatter(yr_te, pred_r, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, alpha=.9, zorder=3)
a1.set_xlim(lo, hi)
a1.set_ylim(lo, hi)
a1.set_xlabel("Actual disease progression")
a1.set_ylabel("Predicted")
a1.grid(axis="x", visible=True)
a1.text(hi - 6, lo + 12, "perfect prediction line", ha="right", fontsize=9.5, color=MUTED)
a1.set_title("Test set: predicted vs actual", loc="left", fontsize=11.5, pad=10)
cs = coef
a2.barh(cs.index, cs.values, color=[BLUE if v > 0 else RED for v in cs.values], height=0.6,
        edgecolor=SURFACE, linewidth=2)
a2.axvline(0, color=BASE, lw=1)
a2.grid(axis="y", visible=False)
a2.grid(axis="x", visible=True)
a2.tick_params(axis="y", length=0, colors=INK2)
a2.set_xlabel("Coefficient (effect of a 1-SD change in the feature)")
a2.set_title("What the model learned", loc="left", fontsize=11.5, pad=10)
header(fig, f"Linear regression explains about {lr['r2'] * 100:.0f}% of the variation on unseen patients",
       f"Test R² {lr['r2']:.2f} (5-fold CV {lr['cv_r2_mean']:.2f} ± {lr['cv_r2_std']:.2f}); typical error "
       f"±{lr['mae']:.0f} points on a 25–346 scale")
legend_row(fig, [("Raises predicted progression", BLUE), ("Lowers it", RED)])
r_s1s2 = Xr["s1"].corr(Xr["s2"])
fig.text(0.07, 0.025, f"Caution: s1 and s2 are blood-serum measures correlated at r = {r_s1s2:.2f}, so their large "
         "opposite coefficients should not be read one at a time.", fontsize=9.5, color=INK2)
fig.savefig(OUT / "04_linear_regression.png", dpi=160)
plt.close(fig)

print("Charts written to", OUT.resolve())
