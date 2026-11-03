"""
Prediction logic for the web app, kept free of any UI code so it can be tested on its own.
"""
import json
import math
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).parent
MALIGNANT, BENIGN = 0, 1          # class codes used by the dataset


def slider_step(lo: float, hi: float) -> float:
    """A round step that gives roughly 200 positions across the range."""
    return float(f"{(hi - lo) / 200:.1g}")


def decimals_for(step: float) -> int:
    return max(0, -math.floor(math.log10(step)))


class Predictor:
    def __init__(self, directory: Path = HERE):
        self.model = joblib.load(directory / "model.joblib")
        self.meta = json.loads((directory / "model_meta.json").read_text())
        self.features = self.meta["features"]
        self.ranges = self.meta["ranges"]

    # ---- slider helpers ------------------------------------------------------
    def slider_spec(self, feature: str) -> dict:
        r = self.ranges[feature]
        step = slider_step(r["min"], r["max"])
        d = decimals_for(step)
        return {"min": round(r["min"], d), "max": round(r["max"], d), "step": step,
                "format": f"%.{d}f"}

    def snap(self, feature: str, value: float) -> float:
        spec = self.slider_spec(feature)
        return round(min(max(value, spec["min"]), spec["max"]), decimals_for(spec["step"]))

    def defaults(self) -> dict:
        return {f: self.snap(f, self.ranges[f]["median"]) for f in self.features}

    def preset(self, kind: str) -> dict:
        """Typical (median) measurements of a real malignant or benign tumour from the data."""
        key = {"malignant": "malignant_median", "benign": "benign_median"}[kind]
        return {f: self.snap(f, self.ranges[f][key]) for f in self.features}

    # ---- prediction ------------------------------------------------------------
    def predict(self, values: dict) -> dict:
        x = pd.DataFrame([[values[f] for f in self.features]], columns=self.features)
        proba = self.model.predict_proba(x)[0]
        p_mal = float(proba[list(self.model.classes_).index(MALIGNANT)])
        label = "malignant" if int(self.model.predict(x)[0]) == MALIGNANT else "benign"
        return {"label": label, "p_malignant": p_mal, "p_benign": 1.0 - p_mal,
                "rules": self.explain(x)}

    def explain(self, x: pd.DataFrame) -> list:
        """The yes/no questions the tree asked on the way to its answer."""
        tree = self.model.tree_
        rules = []
        for node in self.model.decision_path(x).indices:
            if tree.children_left[node] == tree.children_right[node]:
                continue                                   # leaf: no question asked
            feature = self.features[tree.feature[node]]
            threshold = float(tree.threshold[node])
            value = float(x.iloc[0][feature])
            rules.append({"feature": feature, "value": value, "threshold": threshold,
                          "went_left": value <= threshold})
        return rules

    def comparison(self, values: dict) -> list:
        """Your inputs next to the typical benign and malignant values."""
        return [{"Measurement": f, "Your value": values[f],
                 "Typical benign": round(self.ranges[f]["benign_median"], 4),
                 "Typical malignant": round(self.ranges[f]["malignant_median"], 4)}
                for f in self.features]
