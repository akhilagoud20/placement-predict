# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv("raw_placement_data.csv")

# 3. Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# 4. Display last 5 rows
print("\n========== LAST 5 ROWS ==========")
print(df.tail())

# 5. Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 6. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 7. Dataset information
print("\n========== DATASET INFO ==========")
print(df.info())

# 8. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 9. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 10. Missing value percentage
print("\n========== MISSING VALUE PERCENTAGE ==========")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# 11. Duplicate values
print("\n========== DUPLICATES ==========")
print("Number of duplicate rows:", df.duplicated().sum())

# 12. Remove duplicate rows
df = df.drop_duplicates()

# 13. Unique values in each column
print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print("\n", column)
    print(df[column].unique())

# 14. Number of unique values
print("\n========== NUMBER OF UNIQUE VALUES ==========")
print(df.nunique())

# 15. Value counts for categorical columns
print("\n========== CATEGORICAL VALUE COUNTS ==========")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print("\n---", column, "---")
    print(df[column].value_counts())

# 16. Correlation matrix
print("\n========== CORRELATION MATRIX ==========")

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

print(correlation)

# 17. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 18. Histograms for numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    plt.figure(figsize=(7, 5))

    plt.hist(df[column].dropna(), bins=20)

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# 19. Boxplots for numerical columns
for column in numeric_columns:
    plt.figure(figsize=(7, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

# 20. Placement status distribution
if "placement_status" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="placement_status"
    )

    plt.title("Placement Status Distribution")
    plt.xlabel("Placement Status")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

# 21. CGPA vs Salary
if "cgpa" in df.columns and "salary_package_lpa" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="cgpa",
        y="salary_package_lpa"
    )

    plt.title("CGPA vs Salary Package")
    plt.xlabel("CGPA")
    plt.ylabel("Salary Package (LPA)")

    plt.tight_layout()
    plt.show()

# 22. Coding skills vs placement
if "coding_skills" in df.columns and "placement_status" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="placement_status",
        y="coding_skills"
    )

    plt.title("Coding Skills vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Coding Skills")

    plt.tight_layout()
    plt.show()

# 23. Internships vs placement
if "internships" in df.columns and "placement_status" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="internships",
        hue="placement_status"
    )

    plt.title("Internships vs Placement Status")
    plt.xlabel("Number of Internships")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

# 24. Projects vs placement
if "projects_count" in df.columns and "placement_status" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="placement_status",
        y="projects_count"
    )

    plt.title("Projects Count vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Number of Projects")

    plt.tight_layout()
    plt.show()

# 25. Final dataset information
print("\n========== FINAL DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nEDA COMPLETED SUCCESSFULLY!")