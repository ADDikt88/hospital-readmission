"""
Feature Engineering Script
Healthcare Readmission Prediction

Purpose:
- Create clinically motivated, leakage-safe features
- Preserve interpretability for baseline models
- Produce a clean modeling dataset

Author: Kyle Tsang
"""

import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------

CLEAN_DATA_PATH = Path("data/processed/diabetes_clean.csv")
OUTPUT_PATH = Path("data/processed/modeling_dataset1.csv")

AGE_BINS = [0, 40, 65, 80, np.inf]
AGE_LABELS = ["<40", "40-64", "65-79", "80+"]

LOS_BINS = [0, 2, 5, np.inf]
LOS_LABELS = ["0-2", "3-5", "6+"]

RANDOM_SEED = 42


# -----------------------------
# Utility Functions
# -----------------------------

def safe_log_transform(series):
    """
    Apply log1p transform to avoid issues with zero values.
    """
    return np.log1p(series.clip(lower=0))

def age_bin_to_midpoint(age_bin):
    if pd.isna(age_bin):
        return np.nan
    age_bin = age_bin.strip("[]()")
    low, high = age_bin.split("-")
    return (int(low) + int(high)) / 2


# -----------------------------
# Feature Engineering Steps
# -----------------------------

def engineer_demographics(df):
    """
    Demographic features
    """
    df["age_mid"] = df["age"].apply(age_bin_to_midpoint)

    df["age_group"] = pd.cut(
        df["age_mid"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False
    )

    return df

def engineer_length_of_stay(df):
    """
    Length-of-stay features
    """
    df["los_log"] = safe_log_transform(df["time_in_hospital"])

    df["los_group"] = pd.cut(
        df["time_in_hospital"],
        bins=LOS_BINS,
        labels=LOS_LABELS,
        right=False
    )

    return df

def engineer_utilization_history(df):
    """
    Prior utilization features
    NOTE: Ensure these are computed strictly before index admission.
    """
    df["prior_outpatient"] = (df["number_outpatient"] > 0).astype(int)
    df["prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

    return df

def engineer_drug_count(df):
    """
    Aggregate drug features
    """
    DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

    df["num_active_drugs"] = (df[DRUG_COLUMNS] != "No").sum(axis=1)

    return df

# Need to engineer discharge features with mapping file
"""
def engineer_discharge_features(df):
    df["discharge_home"] = (df["discharge_disposition"] == "Home").astype(int)
    df["discharge_facility"] = (
        df["discharge_disposition"].isin(["Skilled Nursing Facility", "Rehab"])
    ).astype(int)

    return df
"""

def handle_missing_values(df):
    """
    Impute missing values conservatively.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype(str).replace('nan', 'Unknown')

    return df


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("Loading data...")
    df = pd.read_csv(CLEAN_DATA_PATH)

    print("Engineering demographics...")
    df = engineer_demographics(df)

    print("Engineering length of stay...")
    df = engineer_length_of_stay(df)

    print("Engineering utilization history...")
    df = engineer_utilization_history(df)

    print("Engineering drug count...")
    df = engineer_drug_count(df)

    #print("Engineering discharge features...")
    #df = engineer_discharge_features(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Saving modeling dataset...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering complete.")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
