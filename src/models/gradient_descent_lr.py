# LAB 5
# Linear Regression using Gradient Descent
# Pure NumPy Implementation
# Learning Rate Experimentation
# 80/20 Train-Test Split
# Cost History Plot
# Comparison with Scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# STEP 1: Create Dataset
# ============================================================

# Hours studied
X = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20
], dtype=float)

# Marks obtained
y = np.array([
    35, 40, 45, 50, 55,
    60, 65, 70, 75, 80,
    82, 85, 88, 90, 92,
    94, 95, 96, 98, 100
], dtype=float)

# Convert X into 2D array
X = X.reshape(-1, 1)


# ============================================================
# STEP 2: 80/20 Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("==========================================")
print("80/20 TRAIN-TEST SPLIT")
print("==========================================")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# STEP 3: Standardize Training Data
# ============================================================

mean_X = np.mean(X_train)
std_X = np.std(X_train)

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X


# ============================================================
# STEP 4: Add Bias Column
# ============================================================

# Model:
# y = w0 + w1*x

X_train_b = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_b = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]


# ============================================================
# STEP 5: Cost Function
# ============================================================

def compute_cost(X, y, weights):

    predictions = X @ weights

    error = predictions - y

    cost = (1 / (2 * len(y))) * np.sum(error ** 2)

    return cost


# ============================================================
# STEP 6: Gradient Descent
# ============================================================

def gradient_descent(X, y, learning_rate, iterations):

    # Initialize weights
    weights = np.zeros(X.shape[1])

    # Store cost for every iteration
    cost_history = []

    for i in range(iterations):

        # Prediction
        predictions = X @ weights

        # Error
        error = predictions - y

        # Gradient
        gradient = (1 / len(y)) * (X.T @ error)

        # Update weights
        weights = weights - learning_rate * gradient

        # Calculate cost
        cost = compute_cost(X, y, weights)

        cost_history.append(cost)

    return weights, cost_history


# ============================================================
# STEP 7: Learning Rate Experimentation
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

    # Prediction on test data
    predictions = X_test_b @ weights

    # Calculate performance
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[lr] = {
        "weights": weights,
        "cost_history": cost_history,
        "mse": mse,
        "r2": r2
    }

    print("\n------------------------------------------")
    print("Learning Rate:", lr)
    print("Weights:", weights)
    print("MSE:", mse)
    print("R2 Score:", r2)


# ============================================================
# STEP 8: Plot Cost History
# ============================================================

plt.figure(figsize=(8, 5))

for lr in learning_rates:

    plt.plot(
        results[lr]["cost_history"],
        label="Learning Rate = " + str(lr)
    )

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History for Different Learning Rates")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# STEP 9: Select Best Learning Rate
# ============================================================

best_lr = min(
    learning_rates,
    key=lambda lr: results[lr]["mse"]
)

best_weights = results[best_lr]["weights"]

print("\n==========================================")
print("BEST GRADIENT DESCENT MODEL")
print("==========================================")

print("Best Learning Rate:", best_lr)
print("Best Weights:", best_weights)


# ============================================================
# STEP 10: NumPy Gradient Descent Predictions
# ============================================================

numpy_predictions = X_test_b @ best_weights

print("\n==========================================")
print("NUMPY GRADIENT DESCENT RESULTS")
print("==========================================")

print("Actual Values:")
print(y_test)

print("\nPredicted Values:")
print(numpy_predictions)

numpy_mse = mean_squared_error(
    y_test,
    numpy_predictions
)

numpy_r2 = r2_score(
    y_test,
    numpy_predictions
)

print("\nMSE:", numpy_mse)
print("R2 Score:", numpy_r2)


# ============================================================
# STEP 11: Scikit-learn Linear Regression
# ============================================================

sk_model = LinearRegression()

sk_model.fit(
    X_train,
    y_train
)

sk_predictions = sk_model.predict(X_test)


# ============================================================
# STEP 12: Scikit-learn Results
# ============================================================

sk_mse = mean_squared_error(
    y_test,
    sk_predictions
)

sk_r2 = r2_score(
    y_test,
    sk_predictions
)

print("\n==========================================")
print("SCIKIT-LEARN LINEAR REGRESSION")
print("==========================================")

print("Coefficient:", sk_model.coef_[0])
print("Intercept:", sk_model.intercept_)

print("MSE:", sk_mse)
print("R2 Score:", sk_r2)


# ============================================================
# STEP 13: Model Comparison
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print("\nNumPy Gradient Descent")
print("MSE      :", numpy_mse)
print("R2 Score :", numpy_r2)

print("\nScikit-learn")
print("MSE      :", sk_mse)
print("R2 Score :", sk_r2)


# ============================================================
# STEP 14: Plot Regression Comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Data"
)

# Sort values for proper line plotting
sort_index = np.argsort(X_test[:, 0])

X_sorted = X_test[sort_index]

numpy_sorted = numpy_predictions[sort_index]
sk_sorted = sk_predictions[sort_index]

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
plt.title("Linear Regression: NumPy vs Scikit-learn")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# STEP 15: Final Summary
# ============================================================

print("\n==========================================")
print("FINAL SUMMARY")
print("==========================================")

print("Best Learning Rate:", best_lr)

print("\nNumPy Gradient Descent:")
print("MSE:", numpy_mse)
print("R2 :", numpy_r2)

print("\nScikit-learn:")
print("MSE:", sk_mse)
print("R2 :", sk_r2)

print("\nLab 5 completed successfully!")