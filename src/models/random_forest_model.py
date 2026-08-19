import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
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
# 2. LOAD DATA
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


# ==========================================
# 3. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 5. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n====================================")
print("RANDOM FOREST EVALUATION")
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
    "random_forest_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print("\n====================================")
print("RANDOM FOREST MODEL SAVED!")
print("====================================")

print("\nSaved at:")
print(model_path)