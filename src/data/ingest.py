import os
import pandas as pd


def load_and_validate_data(file_path: str) -> pd.DataFrame:

    # Check whether the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )

    # Display progress message
    print(f"Executing secure data extraction from: {file_path}")

    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Required columns for the Placement Predict project
    required_columns = [
        "branch",
        "college_tier",
        "cgpa",eda.py
        
        "backlogs",
        "coding_skills",
        "communication_skills",
        "internships",
        "projects_count",
        "placement_status",
        "salary_package_lpa"
    ]

    # Find missing columns
    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    # Stop if required columns are missing
    if missing_cols:
        raise ValueError(
            f"Schema Validation Failure: Missing essential feature targets: {missing_cols}"
        )

    # Display dataset dimensions
    print(
        f"Data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, {df.shape[1]} metrics."
    )

    # Return the DataFrame
    return df


# Execute this block only when this file is run directly
if __name__ == "__main__":

    # Path to the raw dataset
    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    try:
        # Load and validate the dataset
        raw_data = load_and_validate_data(DATA_PATH)

    except Exception as e:
        # Display error message
        print(f"Ingestion lifecycle termination: {str(e)}")