import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 1. Generate sample data
np.random.seed(42)

X = np.sort(6 * np.random.rand(100, 1) + 4)

y = np.sin(X).ravel() + np.random.normal(
    0, 0.2, X.shape[0]
)


# 2. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 3. Convert features into polynomial features
poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 4. Scale the polynomial features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


# 5. Define Ridge regularization values
lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []


# 6. Train Ridge models for different lambda values
for lam in lambdas:

    ridge = Ridge(alpha=lam)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    # Calculate Mean Squared Error
    train_errors.append(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_errors.append(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


# 7. Plot training and testing errors
plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error",
    linewidth=2,
    linestyle="--"
)

plt.xscale("log")

plt.xlabel(
    "Regularization Parameter (Lambda / Alpha)"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Ridge Regression Regularization - Degree 15"
)

plt.legend()

plt.grid(True, which="both", linestyle="--")

plt.show()