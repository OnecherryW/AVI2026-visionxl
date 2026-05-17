import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLS = ("H_self", "E_self", "A_self", "C_self")


def _center_col(col_name: str) -> str:
    return f"{col_name}_centered"


def encode_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    missing = [col for col in RAW_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 1) Clip to [1, 5] for stability
    for col in RAW_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(1.0, 5.0)

    # 2) Center to [-1, 1]
    centered_cols = []
    for col in RAW_COLS:
        centered_name = _center_col(col)
        df[centered_name] = (df[col] - 3.0) / 2.0
        centered_cols.append(centered_name)

    # 3) Summary stats
    df["self_mean"] = df[centered_cols].mean(axis=1)
    df["self_std"] = df[centered_cols].std(axis=1).fillna(0.0)
    df["self_max"] = df[centered_cols].max(axis=1)
    df["self_min"] = df[centered_cols].min(axis=1)
    df["self_range"] = df["self_max"] - df["self_min"]

    # 4) Pairwise deltas
    df["H_minus_E"] = df[_center_col("H_self")] - df[_center_col("E_self")]
    df["H_minus_A"] = df[_center_col("H_self")] - df[_center_col("A_self")]
    df["H_minus_C"] = df[_center_col("H_self")] - df[_center_col("C_self")]
    df["E_minus_A"] = df[_center_col("E_self")] - df[_center_col("A_self")]
    df["E_minus_C"] = df[_center_col("E_self")] - df[_center_col("C_self")]
    df["A_minus_C"] = df[_center_col("A_self")] - df[_center_col("C_self")]

    metadata_cols = [
        _center_col("H_self"),
        _center_col("E_self"),
        _center_col("A_self"),
        _center_col("C_self"),
        "self_mean",
        "self_std",
        "self_max",
        "self_min",
        "self_range",
        "H_minus_E",
        "H_minus_A",
        "H_minus_C",
        "E_minus_A",
        "E_minus_C",
        "A_minus_C",
    ]

    return df, metadata_cols


def encode_csv_file(input_path: Path, output_path: Path) -> list[str]:
    df = pd.read_csv(input_path)
    df_out, metadata_cols = encode_metadata(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return metadata_cols


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode metadata features.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_cols = encode_csv_file(input_path, output_path)
    print("Encoded metadata columns:")
    print(",".join(metadata_cols))


if __name__ == "__main__":
    main()
