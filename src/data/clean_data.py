import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data():
    """Load the raw diabetes readmission dataset."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN."""
    return df.replace("?", np.nan)


def group_icd9(code):
    """
    Map ICD-9 diagnosis codes into clinically meaningful groups.
    (Based on CMS ICD-9 groupings)
    """
    if pd.isna(code):
        return np.nan

    try:
        code = float(code)
    except:
        return np.nan

    if 390 <= code <= 459 or code == 785:
        return "circulatory"
    elif 460 <= code <= 519 or code == 786:
        return "respiratory"
    elif 520 <= code <= 579 or code == 787:
        return "digestive"
    elif 250 <= code < 251:
        return "diabetes"
    elif 800 <= code <= 999:
        return "injury"
    elif 710 <= code <= 739:
        return "musculoskeletal"
    elif 580 <= code <= 629 or code == 788:
        return "genitourinary"
    elif 140 <= code <= 239:
        return "neoplasms"
    else:
        return "other"


def group_diagnoses(df):
    """Group the three diagnosis columns."""
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(group_icd9)
    return df


def encode_readmission(df: pd.DataFrame, mode="binary") -> pd.DataFrame:
    """
    Convert readmission column into:
    - binary: 1 = readmitted (<30 or >30), 0 = no readmission
    - multi_class: keep {NO, <30, >30}
    """
    if mode == "binary":
        df["readmitted_flag"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)
    elif mode == "multi_class":
        df["readmitted_flag"] = df["readmitted"]
    else:
        raise ValueError("mode must be 'binary' or 'multi_class'")
        
    return df


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Certain columns have extremely high missing or are identifiers.
    Remove columns that are not useful for prediction.
    """
    drop_cols = [
        "weight",
        "payer_code",
        "medical_specialty",
        #"encounter_id",
        #"patient_nbr",
    ]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Avoid one-hot encoding the original readmitted label
    if "readmitted" in categorical_cols:
        categorical_cols.remove("readmitted")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def clean_data(save=True, readmission_mode="binary"):
    """
    Master function that executes the full cleaning pipeline.
    """
    print("Loading raw dataset...")
    df = load_raw_data()

    print("Replacing placeholder missing values...")
    df = replace_missing(df)

    print("Grouping ICD-9 diagnosis codes...")
    df = group_diagnoses(df)

    print("Encoding readmission outcome...")
    df = encode_readmission(df, mode=readmission_mode)

    print("Dropping sparse/unnecessary columns...")
    df = drop_sparse_columns(df)

    print("One-hot encoding categorical features...")
    df = encode_categoricals(df)

    if save:
        output_path = PROCESSED_DIR / "diabetes_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    df_clean = clean_data()
    print("Clean dataset preview:")
    print(df_clean.head())
    print(df_clean.shape)
