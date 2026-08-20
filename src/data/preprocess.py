import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "raw_placement_data.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

# student_id is only an identifier.
# salary_package_lpa should not be used to predict
# placement_status because salary is known after placement.

columns_to_drop = [
    "student_id",
    "salary_package_lpa"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ==========================================
# 3. DEFINE TARGET AND FEATURES
# ==========================================

TARGET = "placement_status"

X = df.drop(columns=[TARGET])
y = df[TARGET]


print("\nFeatures:", X.shape)
print("Target:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==========================================
# 5. PREPROCESSING PIPELINE
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 7. FIT PREPROCESSOR
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed training data:", X_train_processed.shape)
print("Processed testing data:", X_test_processed.shape)


# ==========================================
# 8. CREATE OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================

X_train_df = pd.DataFrame(X_train_processed)
X_test_df = pd.DataFrame(X_test_processed)

X_train_df.to_csv(
    os.path.join(OUTPUT_DIR, "X_train.csv"),
    index=False
)

X_test_df.to_csv(
    os.path.join(OUTPUT_DIR, "X_test.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(OUTPUT_DIR, "y_train.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(OUTPUT_DIR, "y_test.csv"),
    index=False
)


# ==========================================
# 10. SAVE PREPROCESSOR
# ==========================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)


# ==========================================
# 11. COMPLETION MESSAGE
# ==========================================

print("\n====================================")
print("DATA PREPROCESSING COMPLETED!")
print("====================================")

print("\nSaved files:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")
print("preprocessor.pkl")