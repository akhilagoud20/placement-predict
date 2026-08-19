import os
import pandas as pd
import joblib

from xgboost import XGBClassifier
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


# Convert target labels to 0 and 1
y_train = y_train.map({
    "Not Placed": 0,
    "Placed": 1
})

y_test = y_test.map({
    "Not Placed": 0,
    "Placed": 1
})


print("Data loaded successfully!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ==========================================
# 3. CREATE XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# ==========================================
# 4. TRAIN
# ==========================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 5. PREDICT
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. EVALUATE
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n====================================")
print("XGBOOST EVALUATION")
print("====================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Placed", "Placed"]
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
    "xgboost_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print("\n====================================")
print("XGBOOST MODEL SAVED!")
print("====================================")

print("\nSaved at:")
print(model_path)