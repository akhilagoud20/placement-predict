from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# Features and target
F = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

T = "placement_status"


def load():
    df = pd.read_csv(DATA)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Select required columns and remove missing values
    df = df[F + [T]].dropna()

    # Convert college tier to numeric
    # Example: Tier 1 -> 1
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[F],
        columns=["branch"],
        dtype=float
    )

    # Encode target
    y = df[T]

    if not pd.api.types.is_numeric_dtype(y):
        lab = sorted(y.astype(str).unique())

        y = y.astype(str).map({
            lab[0]: 0,
            lab[1]: 1
        })

    return X, y


def score(model, X, y):
    predictions = model.predict(X)

    return [
        round(accuracy_score(y, predictions), 4),
        round(precision_score(y, predictions, zero_division=0), 4),
        round(recall_score(y, predictions, zero_division=0), 4),
        round(f1_score(y, predictions, zero_division=0), 4)
    ]


def run():

    # Load data
    X, y = load()

    print("=" * 60)
    print("EXPERIMENT 9 - RANDOM FOREST CLASSIFICATION")
    print("=" * 60)

    print("\nDataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    # Train-test split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTraining samples:", len(Xtr))
    print("Testing samples:", len(Xte))

    # ---------------------------------------------------------
    # 1. Decision Tree vs Random Forest
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("1. DECISION TREE VS RANDOM FOREST")
    print("=" * 60)

    # Baseline Decision Tree
    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(Xtr, ytr)

    dt_scores = score(dt, Xte, yte)

    print(
        "Decision Tree [accuracy, precision, recall, F1]:",
        dt_scores
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(Xtr, ytr)

    rf_scores = score(rf, Xte, yte)

    print(
        "Random Forest [accuracy, precision, recall, F1]:",
        rf_scores
    )

    print(
        "OOB score:",
        round(rf.oob_score_, 4),
        "| OOB error:",
        round(1 - rf.oob_score_, 4)
    )

    # ---------------------------------------------------------
    # 2. Effect of Number of Trees
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. EFFECT OF NUMBER OF TREES")
    print("=" * 60)

    counts = [10, 25, 50, 100, 200]

    oob_errors = []
    accuracies = []

    for n in counts:

        model = RandomForestClassifier(
            n_estimators=n,
            max_features="sqrt",
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        oob_error = 1 - model.oob_score_

        test_accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        oob_errors.append(oob_error)
        accuracies.append(test_accuracy)

        print(
            f"Trees: {n:3d} | "
            f"OOB Error: {oob_error:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

    # Plot
    plt.figure(figsize=(9, 6))

    plt.plot(
        counts,
        oob_errors,
        marker="o",
        label="OOB Error"
    )

    plt.plot(
        counts,
        accuracies,
        marker="o",
        label="Test Accuracy"
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Value")
    plt.title(
        "Effect of Number of Trees - Placement Prediction"
    )

    plt.legend()
    plt.tight_layout()

    tree_plot = OUT / "random_forest_number_of_trees.png"

    plt.savefig(
        tree_plot,
        dpi=150
    )

    plt.close()

    print("\nSaved:", tree_plot)

    # ---------------------------------------------------------
    # 3. Effect of Feature Subsampling
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("3. EFFECT OF FEATURE SUBSAMPLING")
    print("=" * 60)

    options = ["sqrt", "log2", None]

    labels = [
        "sqrt",
        "log2",
        "all features"
    ]

    values = []

    for option in options:

        model = RandomForestClassifier(
            n_estimators=100,
            max_features=option,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        values.append(accuracy)

    print(
        "Feature subsampling:",
        {
            label: round(value, 4)
            for label, value in zip(labels, values)
        }
    )

    # Plot
    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Test Accuracy")
    plt.title(
        "Feature Subsampling - Placement Prediction"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()

    feature_plot = (
        OUT / "random_forest_feature_subsampling.png"
    )

    plt.savefig(
        feature_plot,
        dpi=150
    )

    plt.close()

    print("\nSaved:", feature_plot)

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT 9 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")
    print("1. random_forest_number_of_trees.png")
    print("2. random_forest_feature_subsampling.png")


if __name__ == "__main__":
    run()