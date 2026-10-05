from pathlib import Path

import os

# Limit numerical libraries to one thread to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


def load_data():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier to numbers
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=np.float32
    )

    # Convert target to 0 and 1
    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError(
                "placement_status must contain exactly two classes"
            )

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    y = y.astype(np.int8)

    return X, y


def run():

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    X, y = load_data()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Dataset loaded successfully.")
    print("Training samples:", len(Xtr))
    print("Testing samples:", len(Xte))


    # =========================================================
    # 2. GINI AND ENTROPY
    # =========================================================

    for name, criterion in [
        ("gini", "gini"),
        ("entropy", "entropy")
    ]:

        model = DecisionTreeClassifier(
            criterion=criterion,
            random_state=42
        )

        model.fit(Xtr, ytr)

        predictions = model.predict(Xte)

        accuracy = accuracy_score(
            yte,
            predictions
        )

        print(
            name,
            "test accuracy:",
            round(accuracy, 4)
        )

        plt.figure(figsize=(18, 10))

        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=["0", "1"],
            filled=True,
            max_depth=4,
            fontsize=7
        )

        plt.title(
            f"Placement Decision Tree - {name.title()}"
        )

        plt.tight_layout()

        plt.savefig(
            OUT / f"decision_tree_{name}.png",
            dpi=150
        )

        plt.close()


    # =========================================================
    # 3. COST COMPLEXITY PRUNING
    # =========================================================

    print("\nStarting Cost Complexity Pruning...")

    # Use a smaller stratified sample only for pruning.
    # This avoids excessive memory consumption.
    PRUNING_SIZE = 10000

    X_prune, _, y_prune, _ = train_test_split(
        Xtr,
        ytr,
        train_size=PRUNING_SIZE,
        stratify=ytr,
        random_state=42
    )

    print(
        "Pruning analysis samples:",
        len(X_prune)
    )

    base_tree = DecisionTreeClassifier(
        random_state=42
    )

    path = base_tree.cost_complexity_pruning_path(
        X_prune,
        y_prune
    )

    alphas = np.unique(path.ccp_alphas)

    # Remove the final alpha that produces a root-only tree
    if len(alphas) > 1:
        alphas = alphas[:-1]

    # Keep at most 20 representative values
    if len(alphas) > 20:

        indices = np.linspace(
            0,
            len(alphas) - 1,
            20,
            dtype=int
        )

        alphas = alphas[indices]

    pruning_results = []

    for i, alpha in enumerate(alphas):

        print(
            f"Pruning model {i + 1}/{len(alphas)}"
        )

        model = DecisionTreeClassifier(
            ccp_alpha=float(alpha),
            random_state=42
        )

        model.fit(
            X_prune,
            y_prune
        )

        train_accuracy = accuracy_score(
            y_prune,
            model.predict(X_prune)
        )

        test_accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        pruning_results.append([
            alpha,
            train_accuracy,
            test_accuracy,
            model.tree_.node_count
        ])


    pruning_results = pd.DataFrame(
        pruning_results,
        columns=[
            "alpha",
            "train_accuracy",
            "test_accuracy",
            "nodes"
        ]
    )


    best = pruning_results.loc[
        pruning_results["test_accuracy"].idxmax()
    ]

    print("\nCost Complexity Pruning")

    print(
        "Best CCP alpha:",
        best["alpha"]
    )

    print(
        "Pruned test accuracy:",
        best["test_accuracy"]
    )


    # Save CCP graph
    plt.figure(figsize=(9, 6))

    plt.plot(
        pruning_results["alpha"],
        pruning_results["train_accuracy"],
        marker="o",
        label="Training"
    )

    plt.plot(
        pruning_results["alpha"],
        pruning_results["test_accuracy"],
        marker="o",
        label="Testing"
    )

    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy")

    plt.title(
        "Cost Complexity Pruning - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_ccp.png",
        dpi=150
    )

    plt.close()


    # =========================================================
    # 4. EFFECT OF TREE DEPTH
    # =========================================================

    print("\nStarting Tree Depth Analysis...")

    depths = range(1, 16)

    train_accuracy = []
    test_accuracy = []

    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(
            Xtr,
            ytr
        )

        train_accuracy.append(
            accuracy_score(
                ytr,
                model.predict(Xtr)
            )
        )

        test_accuracy.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    best_depth = list(depths)[
        int(np.argmax(test_accuracy))
    ]

    best_depth_accuracy = max(
        test_accuracy
    )

    print("\nEffect of Tree Depth")

    print(
        "Best depth:",
        best_depth
    )

    print(
        "Best test accuracy:",
        best_depth_accuracy
    )


    # Save depth graph
    plt.figure(figsize=(9, 6))

    plt.plot(
        depths,
        train_accuracy,
        marker="o",
        label="Training"
    )

    plt.plot(
        depths,
        test_accuracy,
        marker="o",
        label="Testing"
    )

    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")

    plt.title(
        "Effect of Tree Depth - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_depth.png",
        dpi=150
    )

    plt.close()


    # =========================================================
    # 5. FINAL
    # =========================================================

    print("\nAll graphs saved successfully.")

    print("\nGenerated files:")
    print("decision_tree_gini.png")
    print("decision_tree_entropy.png")
    print("decision_tree_ccp.png")
    print("decision_tree_depth.png")


if __name__ == "__main__":
    run()