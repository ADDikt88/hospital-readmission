"""
load_data.py

Downloads the Kaggle dataset using kagglehub, organizes it into the
project's data/raw directory, and returns the final file paths.

This script is part of the LOS Prediction project.
"""

import requests
import zipfile
import io
import os
import pandas as pd

raw_dir = "data/raw"


# URL of the UCI dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

# Step 1: Download the ZIP file
response = requests.get(url)
response.raise_for_status()

# Step 2: Extract the ZIP file into memory
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # List files inside the zip (for debugging)
    print("Files in ZIP:", z.namelist())
    
    # The main file is 'diabetic_data.csv'
    with z.open('dataset_diabetes/diabetic_data.csv') as f:
        df = pd.read_csv(f)

    save_path = os.path.join(raw_dir, 'diabetic_data.csv')
    df.to_csv(save_path, index=False)
    print(f"Saved raw data to: {save_path}")

# Step 3: Display the DataFrame
print(df.head())
print(df.shape)

