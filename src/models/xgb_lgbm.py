from pathlib import Path
import os
import time

# Keep memory usage low on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "src" / "data" / "raw_placement_data.csv"

OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count",
]

TARGET = "placement_status"


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    print("Original dataset shape:", df.shape)

    # Keep only required columns
    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier such as "Tier 1" -> 1
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=np.float32
    )

    # Convert target into 0 and 1
    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        classes = sorted(y.astype(str).unique())

        if len(classes) != 2:
            raise ValueError(
                f"placement_status must contain exactly 2 classes. "
                f"Found: {classes}"
            )

        mapping = {
            classes[0]: 0,
            classes[1]: 1
        }

        y = y.astype(str).map(mapping)

        print("Target mapping:", mapping)

    y = y.astype(np.int8)

    print("Final feature shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# SHAP ANALYSIS
# ============================================================

def run_shap_analysis(model, X_test, model_name):

    print(f"\nRunning SHAP analysis for {model_name}...")

    # Use a smaller sample to avoid excessive memory usage
    sample_size = min(1000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print("SHAP sample size:", len(X_sample))

    # TreeExplainer works well with XGBoost and LightGBM
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # SHAP output differs between library/model versions
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = shap_values[1]
        else:
            values = shap_values[0]

    elif len(getattr(shap_values, "shape", ())) == 3:

        # Some versions return:
        # samples x features x classes
        values = shap_values[:, :, 1]

    else:

        values = shap_values

    values = np.asarray(values)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    print("\nTop influential features:")

    print(
        importance.head(10).to_string(index=False)
    )

    # Save feature importance
    importance_file = (
        OUTPUT_DIR /
        f"{model_name}_shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    print(
        "Saved:",
        importance_file
    )

    # ========================================================
    # SHAP SUMMARY PLOT
    # ========================================================

    plt.close("all")

    shap.summary_plot(
        values,
        X_sample,
        show=False
    )

    summary_file = (
        OUTPUT_DIR /
        f"{model_name}_shap_summary.png"
    )

    plt.tight_layout()

    plt.savefig(
        summary_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close("all")

    print(
        "Saved:",
        summary_file
    )

    # ========================================================
    # SHAP DEPENDENCE PLOT
    # ========================================================

    top_feature = importance.iloc[0]["Feature"]

    print(
        "Top feature for dependence plot:",
        top_feature
    )

    plt.close("all")

    shap.dependence_plot(
        top_feature,
        values,
        X_sample,
        show=False
    )

    dependence_file = (
        OUTPUT_DIR /
        f"{model_name}_shap_dependence.png"
    )

    plt.tight_layout()

    plt.savefig(
        dependence_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close("all")

    print(
        "Saved:",
        dependence_file
    )

    return importance


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EXPERIMENT 8")
    print("XGBoost and LightGBM Placement Prediction")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ========================================================
    # XGBOOST
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    )

    start_time = time.perf_counter()

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_time = time.perf_counter() - start_time

    xgb_pred = xgb_model.predict(X_test)

    xgb_accuracy = accuracy_score(
        y_test,
        xgb_pred
    )

    print("\nXGBoost Training Time:",
          round(xgb_time, 4), "seconds")

    print(
        "XGBoost Accuracy:",
        round(xgb_accuracy, 4)
    )

    print("\nXGBoost Classification Report:")

    print(
        classification_report(
            y_test,
            xgb_pred,
            zero_division=0
        )
    )

    # SHAP
    xgb_importance = run_shap_analysis(
        xgb_model,
        X_test,
        "xgboost"
    )

    # ========================================================
    # LIGHTGBM
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM")
    print("=" * 60)

    lgb_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=1
    )

    start_time = time.perf_counter()

    lgb_model.fit(
        X_train,
        y_train
    )

    lgb_time = time.perf_counter() - start_time

    lgb_pred = lgb_model.predict(X_test)

    lgb_accuracy = accuracy_score(
        y_test,
        lgb_pred
    )

    print("\nLightGBM Training Time:",
          round(lgb_time, 4), "seconds")

    print(
        "LightGBM Accuracy:",
        round(lgb_accuracy, 4)
    )

    print("\nLightGBM Classification Report:")

    print(
        classification_report(
            y_test,
            lgb_pred,
            zero_division=0
        )
    )

    # SHAP
    lgb_importance = run_shap_analysis(
        lgb_model,
        X_test,
        "lightgbm"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = pd.DataFrame({
        "Model": [
            "XGBoost",
            "LightGBM"
        ],
        "Accuracy": [
            xgb_accuracy,
            lgb_accuracy
        ],
        "Training_Time_Seconds": [
            xgb_time,
            lgb_time
        ]
    })

    comparison_file = (
        OUTPUT_DIR /
        "xgb_lightgbm_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(index=False)
    )

    print(
        "\nSaved comparison:",
        comparison_file
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("EXPERIMENT 8 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print("1. xgboost_shap_feature_importance.csv")
    print("2. xgboost_shap_summary.png")
    print("3. xgboost_shap_dependence.png")
    print("4. lightgbm_shap_feature_importance.csv")
    print("5. lightgbm_shap_summary.png")
    print("6. lightgbm_shap_dependence.png")
    print("7. xgb_lightgbm_comparison.csv")


if __name__ == "__main__":
    main()