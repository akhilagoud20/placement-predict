import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()


print("Data loaded successfully!")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ==========================================
# 3. CREATE MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 7. SAVE MODEL
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "placement_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n====================================")
print("MODEL SAVED SUCCESSFULLY!")
print("====================================")

print("\nSaved at:")
print(model_path)