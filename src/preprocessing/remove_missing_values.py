"""
remove_missing_values.py

This script removes all rows from a CSV file where any required (non-timestamp) column contains missing values.
The timestamp column is preserved but excluded from missing value checks.

Modify the input and output file paths below to suit your use case.
"""

import pandas as pd
import os

# USER SETTINGS (EDIT THESE)
INPUT_FILE = "data/filtered_flow.csv"              # Raw or partially cleaned input data
OUTPUT_FILE = "data/training_data_no_missing.csv"  # File to save after removing rows with missing values
TIMESTAMP_COL = "timestamp"                        # Column name to exclude from NaN check
ENCODING = "utf-8"                                 # File encoding
SEP = ";"                                          # CSV separator
DECIMAL = ","                                      # Decimal separator

def remove_missing_values(
    input_file: str,
    output_file: str,
    timestamp_col: str = "timestamp",
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "utf-8"
) -> None:
    """
    Removes rows where any required (non-timestamp) column has missing values.
    """
    df = pd.read_csv(input_file, sep=sep, decimal=decimal, encoding=encoding, engine="python")

    # Determine required columns (exclude timestamp)
    required_cols = [col for col in df.columns if col.lower() != timestamp_col.lower()]

    # Remove rows with missing values in required columns
    cleaned_df = df.dropna(subset=required_cols)

    # Sort by timestamp if possible
    if timestamp_col in cleaned_df.columns:
        try:
            cleaned_df[timestamp_col] = pd.to_datetime(cleaned_df[timestamp_col], errors="coerce")
            cleaned_df = cleaned_df.sort_values(timestamp_col)
        except Exception:
            pass

    # Save cleaned data
    cleaned_df.to_csv(output_file, sep=sep, decimal=decimal, index=False)
    removed = len(df) - len(cleaned_df)

    print(f"Saved cleaned file to: {output_file}")
    print(f" → Removed {removed} rows with missing values.")
    print(f" → Remaining rows: {len(cleaned_df)}")

# Run
if __name__ == "__main__":
    remove_missing_values(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        timestamp_col=TIMESTAMP_COL,
        sep=SEP,
        decimal=DECIMAL,
        encoding=ENCODING
    )
