# LAB 5
# Linear Regression using Gradient Descent
# Lab 5 - 6 Graphs + Scikit-learn Comparison

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# CREATE FOLDER FOR GRAPHS
# ============================================================

os.makedirs("reports/figures", exist_ok=True)


# ============================================================
# STEP 1: DATASET
# ============================================================

X = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20
], dtype=float)

y = np.array([
    35, 40, 45, 50, 55,
    60, 65, 70, 75, 80,
    82, 85, 88, 90, 92,
    94, 95, 96, 98, 100
], dtype=float)

X = X.reshape(-1, 1)


# ============================================================
# STEP 2: 80/20 TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# GRAPH 1: ORIGINAL DATASET
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(X, y)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Original Dataset")
plt.grid()

plt.savefig(
    "reports/figures/graph1_original_dataset.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# STEP 3: STANDARDIZATION
# ============================================================

mean_X = np.mean(X_train)
std_X = np.std(X_train)

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X


# Add bias column

X_train_b = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_b = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]


# ============================================================
# STEP 4: COST FUNCTION
# ============================================================

def compute_cost(X, y, weights):

    predictions = X @ weights

    error = predictions - y

    cost = (1 / (2 * len(y))) * np.sum(error ** 2)

    return cost


# ============================================================
# STEP 5: GRADIENT DESCENT
# ============================================================

def gradient_descent(X, y, learning_rate, iterations):

    weights = np.zeros(X.shape[1])

    cost_history = []

    for i in range(iterations):

        predictions = X @ weights

        error = predictions - y

        gradient = (1 / len(y)) * (X.T @ error)

        weights = weights - learning_rate * gradient

        cost = compute_cost(
            X,
            y,
            weights
        )

        cost_history.append(cost)

    return weights, cost_history


# ============================================================
# STEP 6: LEARNING RATE EXPERIMENTATION
# ============================================================

learning_rates = [0.001, 0.01, 0.1]

iterations = 1000

results = {}


for lr in learning_rates:

    weights, cost_history = gradient_descent(
        X_train_b,
        y_train,
        lr,
        iterations
    )

    predictions = X_test_b @ weights

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[lr] = {
        "weights": weights,
        "cost_history": cost_history,
        "mse": mse,
        "r2": r2
    }

    print("\nLearning Rate:", lr)
    print("MSE:", mse)
    print("R2 Score:", r2)


# ============================================================
# GRAPH 2: LEARNING RATE 0.001
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results[0.001]["cost_history"]
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History - Learning Rate 0.001")
plt.grid()

plt.savefig(
    "reports/figures/graph2_cost_lr_0.001.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 3: LEARNING RATE 0.01
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results[0.01]["cost_history"]
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History - Learning Rate 0.01")
plt.grid()

plt.savefig(
    "reports/figures/graph3_cost_lr_0.01.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 4: LEARNING RATE 0.1
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results[0.1]["cost_history"]
)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History - Learning Rate 0.1")
plt.grid()

plt.savefig(
    "reports/figures/graph4_cost_lr_0.1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# STEP 7: BEST MODEL
# ============================================================

best_lr = min(
    learning_rates,
    key=lambda lr: results[lr]["mse"]
)

best_weights = results[best_lr]["weights"]

numpy_predictions = X_test_b @ best_weights

numpy_mse = mean_squared_error(
    y_test,
    numpy_predictions
)

numpy_r2 = r2_score(
    y_test,
    numpy_predictions
)

print("\nBest Learning Rate:", best_lr)

print("\nNumPy Gradient Descent")
print("MSE:", numpy_mse)
print("R2:", numpy_r2)


# ============================================================
# GRAPH 5: NUMPY GRADIENT DESCENT
# ============================================================

sort_index = np.argsort(
    X_test[:, 0]
)

X_sorted = X_test[sort_index]

numpy_sorted = numpy_predictions[
    sort_index
]


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Data"
)

plt.plot(
    X_sorted,
    numpy_sorted,
    label="Gradient Descent"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")

plt.title(
    "NumPy Gradient Descent Regression"
)

plt.legend()
plt.grid()

plt.savefig(
    "reports/figures/graph5_numpy_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# STEP 8: SCIKIT-LEARN
# ============================================================

sk_model = LinearRegression()

sk_model.fit(
    X_train,
    y_train
)

sk_predictions = sk_model.predict(
    X_test
)

sk_mse = mean_squared_error(
    y_test,
    sk_predictions
)

sk_r2 = r2_score(
    y_test,
    sk_predictions
)


print("\nScikit-learn Linear Regression")

print(
    "Coefficient:",
    sk_model.coef_[0]
)

print(
    "Intercept:",
    sk_model.intercept_
)

print(
    "MSE:",
    sk_mse
)

print(
    "R2:",
    sk_r2
)


# ============================================================
# GRAPH 6: NUMPY VS SCIKIT-LEARN
# ============================================================

numpy_sorted = numpy_predictions[
    sort_index
]

sk_sorted = sk_predictions[
    sort_index
]


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Data"
)

plt.plot(
    X_sorted,
    numpy_sorted,
    label="NumPy Gradient Descent"
)

plt.plot(
    X_sorted,
    sk_sorted,
    linestyle="--",
    label="Scikit-learn"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")

plt.title(
    "NumPy vs Scikit-learn Linear Regression"
)

plt.legend()
plt.grid()

plt.savefig(
    "reports/figures/graph6_numpy_vs_sklearn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==========================================")
print("FINAL SUMMARY")
print("==========================================")

print(
    "Best Learning Rate:",
    best_lr
)

print("\nNumPy Gradient Descent:")

print(
    "MSE:",
    numpy_mse
)

print(
    "R2 :",
    numpy_r2
)

print("\nScikit-learn:")

print(
    "MSE:",
    sk_mse
)

print(
    "R2 :",
    sk_r2
)

print("\n==========================================")
print("Lab 5 completed successfully!")
print("All 6 graphs saved in reports/figures")
print("==========================================")