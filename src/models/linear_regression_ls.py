import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_linear_regression_ls():

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    data_path = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = pd.read_csv(data_path)

    print("=" * 50)
    print("LAB 4 - LINEAR REGRESSION")
    print("STANDARD LEAST SQUARES METHOD")
    print("=" * 50)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------
    # 2. Select Input and Output
    # --------------------------------------------------

    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    target_col = "salary_package_lpa"

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(f"\nLoaded {N} data points")
    print(f"Input dimension L = {X_raw.shape[1]}")
    print(f"Output dimension M = {y.shape[1]}")

    # --------------------------------------------------
    # 3. Create Design Matrix
    # --------------------------------------------------

    # X_design = [1, cgpa, communication_skill_score]

    X_design = np.hstack(
        (
            np.ones((N, 1)),
            X_raw
        )
    )

    # --------------------------------------------------
    # 4. Standard Least Squares Method
    # --------------------------------------------------

    # Formula:
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    try:

        XT_X_inv = np.linalg.inv(
            XT_X
        )

    except np.linalg.LinAlgError:

        print("\nMatrix is singular.")
        print("Using pseudo-inverse.")

        XT_X_inv = np.linalg.pinv(
            XT_X
        )

    XT_y = np.dot(
        X_design.T,
        y
    )

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # --------------------------------------------------
    # 5. Display Parameters
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("--- OPTIMAL MODEL PARAMETERS ---")
    print("=" * 50)

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for cgpa (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for communication_skill_score (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )

    # --------------------------------------------------
    # 6. Predictions
    # --------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # --------------------------------------------------
    # 7. Error Function
    # --------------------------------------------------

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"\nMinimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # --------------------------------------------------
    # 8. 3D Regression Plane
    # --------------------------------------------------

    print("\nGenerating 3D regression plane...")

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Actual data points

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.4,
        label="Actual Data"
    )

    # --------------------------------------------------
    # 9. Create Mesh Grid
    # --------------------------------------------------

    x1 = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    x2 = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1,
        x2
    )

    # --------------------------------------------------
    # 10. Calculate Regression Plane
    # --------------------------------------------------

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3
    )

    # --------------------------------------------------
    # 11. Labels
    # --------------------------------------------------

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Communication Skill Score"
    )

    ax.set_zlabel(
        "Salary Package LPA"
    )

    ax.set_title(
        "Linear Regression - Standard Least Squares"
    )

    # --------------------------------------------------
    # 12. Save Graph
    # --------------------------------------------------

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"\n3D regression plane saved to:\n"
        f"{output_path}"
    )

    # --------------------------------------------------
    # 13. Completion Message
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("LAB 4 EXECUTION COMPLETE")
    print("=" * 50)


# ------------------------------------------------------
# Start the program
# ------------------------------------------------------

if __name__ == "__main__":
    train_linear_regression_ls()