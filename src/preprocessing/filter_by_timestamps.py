"""
filter_by_timestamps.py

This script filters a target CSV file by keeping only the rows with timestamps that exist in a reference file.

Modify the input and output filenames below as needed.
"""

import pandas as pd
import os

# USER-DEFINED FILE PATHS (EDIT THESE)
REFERENCE_FILE = "data/filtered_flow.csv"       # File with valid timestamps
TARGET_FILE = "data/raw_turbidity.csv"          # File to be filtered
OUTPUT_FILE = "data/turbidity_aligned.csv"      # Output path for filtered result

# Settings 
TIMESTAMP_COL = "timestamp"
ENCODING_REF = "utf-8"
ENCODING_TARGET = "utf-8"
SEP = ";"
DECIMAL = ","

def filter_by_timestamps(
    reference_file: str,
    target_file: str,
    output_file: str,
    timestamp_col: str = "timestamp",
    encoding_ref: str = "utf-8",
    encoding_target: str = "utf-8",
    sep: str = ";",
    decimal: str = ","
) -> None:
    """
    Filters the target file by keeping only rows whose timestamp exists in the reference file.
    """
    # Read files
    df_ref = pd.read_csv(reference_file, sep=sep, decimal=decimal,
                         parse_dates=[timestamp_col], encoding=encoding_ref)
    df_target = pd.read_csv(target_file, sep=sep, decimal=decimal,
                            parse_dates=[timestamp_col], encoding=encoding_target)

    # Ensure timestamps are datetime
    df_ref[timestamp_col] = pd.to_datetime(df_ref[timestamp_col])
    df_target[timestamp_col] = pd.to_datetime(df_target[timestamp_col])

    # Filter target by matching timestamps
    filtered_df = df_target[df_target[timestamp_col].isin(df_ref[timestamp_col])]

    # Save filtered file
    filtered_df.to_csv(output_file, sep=sep, decimal=decimal, index=False)
    print(f"Filtered file saved: {output_file} ({len(filtered_df)} rows)")

# Run
if __name__ == "__main__":
    filter_by_timestamps(
        reference_file=REFERENCE_FILE,
        target_file=TARGET_FILE,
        output_file=OUTPUT_FILE,
        timestamp_col=TIMESTAMP_COL,
        encoding_ref=ENCODING_REF,
        encoding_target=ENCODING_TARGET,
        sep=SEP,
        decimal=DECIMAL
    )
