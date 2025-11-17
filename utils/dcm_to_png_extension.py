#!/usr/bin/env python3
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path


def inplace_replace_dcm(input_path: str):
    input_path = Path(input_path).resolve()

    # Read parquet into pandas DataFrame
    df = pq.read_table(input_path).to_pandas()

    columns = ["image_path", "cropped_path", "roi_path"]
    print(f"Processing columns: {columns}")

    replaced = 0
    for col in columns:
        if col in df.columns:
            count_before = df[col].str.contains(".dcm", regex=False).sum()
            df[col] = df[col].str.replace(".dcm", ".png", regex=False)
            replaced += count_before

    # Save back to the same file
    pq.write_table(pa.Table.from_pandas(df), input_path)

    print(f"Replaced '.dcm' → '.png' in {replaced} total entries.")
    print(f"Changes saved in place: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace '.dcm' with '.png' in a Parquet file (in place).")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to the Parquet file.")
    args = parser.parse_args()

    inplace_replace_dcm(args.file)
