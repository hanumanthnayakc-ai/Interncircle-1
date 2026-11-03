"""
Exploratory Data Analysis on the Titanic dataset (Kaggle / datasciencedojo schema).

Usage:
    python titanic_eda.py path/to/titanic.csv

Expected columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp,
                  Parch, Ticket, Fare, Cabin, Embarked
"""
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def survival_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Survival rate and group size for each value of `col`."""
    out = df.groupby(col, observed=True)["Survived"].agg(["mean", "sum", "count"])
    out.columns = ["survival_rate", "survived", "passengers"]
    out["survival_rate"] = (out["survival_rate"] * 100).round(1)
    return out


def main(path: str) -> None:
    df = pd.read_csv(path)

    # ------------------------------------------------------------------ 1
    section("1. STRUCTURE")
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nFirst rows:\n", df.head())

    section("2. MISSING VALUES")
    miss = df.isna().sum()
    miss = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(1)})
    print(miss[miss["missing"] > 0].sort_values("missing", ascending=False))
    print("\nDuplicate rows:", df.duplicated().sum())

    # ------------------------------------------------------------------ 2
    section("3. DESCRIPTIVE STATISTICS")
    print(df[["Age", "Fare", "SibSp", "Parch"]].describe().round(2))
    print("\nSkewness:\n", df[["Age", "Fare", "SibSp", "Parch"]].skew().round(2))
    overall = df["Survived"].mean() * 100
    print(f"\nOverall survival rate: {overall:.1f}%  "
          f"({int(df['Survived'].sum())} of {len(df)})")

    # ------------------------------------------------------------------ 3
    section("4. SURVIVAL BY CATEGORY")
    for col in ["Sex", "Pclass", "Embarked"]:
        print(f"\n-- {col} --")
        print(survival_by(df, col))

    print("\n-- Sex x Pclass (survival %) --")
    print((df.pivot_table(values="Survived", index="Sex", columns="Pclass",
                          aggfunc="mean") * 100).round(1))

    # ------------------------------------------------------------------ 4
    section("5. AGE, FARE AND FAMILY SIZE")
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 60, 120],
        labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)",
                "Middle (36-60)", "Senior (60+)"])
    print("\n-- Age group (known ages only) --")
    print(survival_by(df, "AgeGroup"))

    print("\nMedian age by Pclass:\n", df.groupby("Pclass")["Age"].median())
    print("\nMedian age, survived vs not:\n", df.groupby("Survived")["Age"].median())

    df["FareBand"] = pd.qcut(df["Fare"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    print("\n-- Fare quartile --")
    print(survival_by(df, "FareBand"))
    print("\nMedian fare by Pclass:\n", df.groupby("Pclass")["Fare"].median())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyBand"] = pd.cut(df["FamilySize"], bins=[0, 1, 4, 100],
                              labels=["Alone", "Small (2-4)", "Large (5+)"])
    print("\n-- Family size --")
    print(survival_by(df, "FamilyBand"))

    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    common = df["Title"].value_counts()
    df["Title"] = df["Title"].where(df["Title"].isin(common[common >= 10].index), "Other")
    print("\n-- Title extracted from Name --")
    print(survival_by(df, "Title").sort_values("survival_rate", ascending=False))

    if "Cabin" in df:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        print("\n-- Cabin recorded? --")
        print(survival_by(df, "HasCabin"))

    # ------------------------------------------------------------------ 5
    section("6. CORRELATIONS WITH SURVIVAL")
    num = df.assign(Male=(df["Sex"] == "male").astype(int))[
        ["Survived", "Pclass", "Male", "Age", "Fare", "SibSp", "Parch", "FamilySize"]]
    corr = num.corr()["Survived"].drop("Survived").sort_values()
    print(corr.round(3))

    # ------------------------------------------------------------------ 6
    section("7. OUTLIERS (IQR rule)")
    for col in ["Age", "Fare"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"{col}: {n} outliers outside [{lo:.1f}, {hi:.1f}]")

    # Optional charts, if matplotlib is installed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        for a, (col, title) in zip(
            ax.flat,
            [("Sex", "Sex"), ("Pclass", "Class"), ("Embarked", "Port"),
             ("AgeGroup", "Age group"), ("FamilyBand", "Family size")]):
            (df.groupby(col, observed=True)["Survived"].mean() * 100).plot.bar(
                ax=a, color="#4C72B0")
            a.set_title(f"Survival % by {title}")
            a.set_ylabel("%")
            a.set_xlabel("")
            a.tick_params(axis="x", rotation=30)
        df.boxplot(column="Fare", by="Pclass", ax=ax[1, 2])
        ax[1, 2].set_ylim(0, 300)
        ax[1, 2].set_title("Fare by class (clipped at 300)")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig("titanic_eda_charts.png", dpi=120)
        print("\nSaved charts to titanic_eda_charts.png")
    except ImportError:
        print("\n(matplotlib not installed; skipped charts)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python titanic_eda.py path/to/titanic.csv")
    main(sys.argv[1])
