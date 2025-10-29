"""
correct_out_of_bounds.py

This script validates and corrects numeric sensor values so that they remain within predefined limits.
Values outside limits are either replaced with the mean of the previous and next valid value,
or the entire row is removed if correction is not possible.

Modify file paths and limit values below to match your dataset.
"""

import pandas as pd
import numpy as np
import os

# USER-DEFINED SETTINGS (EDIT THESE)
INPUT_FILE = "data/raw_training_data.csv"          # Input data file
OUTPUT_FILE = "data/training_data_corrected.csv"   # Output file for cleaned data

# Column-specific min and max limits
LIMITS = {
    "raaka_sameus": (0.1, 15),
    "tuleva_virtaus": (700, 2200),
    "tuleva_lampotila": (0, 30),
    "alku_pH": (4, 7),
    "kemikaaliannos": (0, 120),
    "flotaatio_sameus": (0.5, 4)
}

TIMESTAMP_COL = "timestamp"     # Name of timestamp column
ENCODING = "utf-8"              # File encoding
SEP = ";"                       # CSV separator
DECIMAL = ","                   # Decimal separator

def correct_out_of_bounds(
    input_file: str,
    output_file: str,
    limits: dict,
    timestamp_col: str = "timestamp",
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "utf-8"
) -> None:
    """
    Corrects out-of-bound values in the given dataset based on provided limits.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    output_file : str
        Path to save the corrected CSV file.
    limits : dict
        Dictionary mapping column names to (min, max) limits.
    timestamp_col : str
        Name of the timestamp column.
    sep, decimal, encoding : str
        CSV formatting settings.
    """

    # Load dataset
    df = pd.read_csv(input_file, sep=sep, decimal=decimal,
                     encoding=encoding, parse_dates=[timestamp_col], dayfirst=True)

    corrected_df = df.copy()
    removed_indices = set()
    corrections = []

    for col, (min_val, max_val) in limits.items():
        if col not in df.columns:
            print(f"[!] Column '{col}' not found, skipping.")
            continue

        series = pd.to_numeric(corrected_df[col], errors='coerce')
        invalid_mask = (series < min_val) | (series > max_val)

        for idx in series[invalid_mask].index:
            if idx == 0 or idx == len(series) - 1:
                removed_indices.add(idx)
                continue

            prev_val = series.iloc[idx - 1]
            next_val = series.iloc[idx + 1]

            # If neighboring values invalid or missing, remove row
            if pd.isna(prev_val) or pd.isna(next_val):
                removed_indices.add(idx)
                continue

            # Replace with average of neighbors if valid
            replacement = (prev_val + next_val) / 2

            if replacement < min_val or replacement > max_val:
                removed_indices.add(idx)
            else:
                original = series.iloc[idx]
                corrected_df.at[idx, col] = replacement
                corrections.append({
                    "timestamp": df.at[idx, timestamp_col],
                    "column": col,
                    "original": original,
                    "replaced_with": replacement
                })

    # Remove invalid rows
    if removed_indices:
        corrected_df = corrected_df.drop(index=list(removed_indices))

    # Convert datatypes safely
    for col in corrected_df.columns:
        if col.lower().startswith(timestamp_col.lower()):
            corrected_df[col] = pd.to_datetime(corrected_df[col], errors="coerce")
        else:
            corrected_df[col] = pd.to_numeric(corrected_df[col], errors="coerce")

    # Save cleaned data
    corrected_df.to_csv(output_file, sep=sep, decimal=decimal, index=False)
    print(f"Cleaned dataset saved: {output_file}")
    print(f" → {len(corrections)} values corrected, {len(removed_indices)} rows removed.")
    
# Run
if __name__ == "__main__":
    correct_out_of_bounds(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        limits=LIMITS,
        timestamp_col=TIMESTAMP_COL,
        sep=SEP,
        decimal=DECIMAL,
        encoding=ENCODING
    )
