"""Validate HydroMLYR source files and build the cross-region model CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


EXPECTED_MD5 = {
    "7_HydroMLYR.zip": "b0e1b8e356849bf30366fb9f0e2c55b3",
    "readme.txt": "c95ad1caf26805f3b86ed471d363a1fe",
    "8_attribute_descriptions.xlsx": "9012868eb5b4dd7be851b4de7a38ae09",
}

MET_COLUMNS = [
    "date",
    "pre",
    "evp",
    "gst_mean",
    "prs_mean",
    "tem_mean",
    "rhu",
    "win_mean",
    "gst_min",
    "prs_min",
    "tem_min",
    "gst_max",
    "prs_max",
    "tem_max",
    "ssd",
    "win_max",
]


def parse_args() -> argparse.Namespace:
    """Build command-line arguments for validation and feature preparation."""

    parser = argparse.ArgumentParser(
        description="Validate HydroMLYR and prepare the Yellow River model CSV."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--zip-name", default="7_HydroMLYR.zip")
    parser.add_argument(
        "--summary-csv", default="hydromlyr_validation_summary.csv"
    )
    parser.add_argument(
        "--model-csv", default="yellow_river_hydromlyr_model_ready.csv"
    )
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the MD5 digest of a file without loading it all into memory."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_from_zip(zip_file: ZipFile, name: str) -> pd.DataFrame:
    """Read one CSV member from the HydroMLYR archive and parse its date."""
    with zip_file.open(name) as f:
        raw = f.read()
    df = pd.read_csv(BytesIO(raw))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def read_text_from_zip(zip_file: ZipFile, name: str) -> str:
    """Read one UTF-8 text member from the HydroMLYR archive."""
    return zip_file.read(name).decode("utf-8", errors="replace")


def basin_ids(zip_file: ZipFile) -> list[str]:
    """Return sorted basin identifiers present in the archive."""
    ids = set()
    prefix = "7_HydroMLYR/1_data/"
    for name in zip_file.namelist():
        if not name.startswith(prefix):
            continue
        parts = name.split("/")
        if len(parts) >= 4 and parts[2]:
            ids.add(parts[2])
    return sorted(ids)


def add_window_features(met: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    """Join streamflow with 7-day and 14-day meteorological aggregates."""
    met = met.sort_values("date").drop_duplicates("date").set_index("date")
    flow = flow.sort_values("date").dropna(subset=["date", "q"]).copy()

    base = pd.DataFrame(index=flow["date"])
    sum_cols = ["pre", "evp", "ssd"]
    mean_cols = ["tem_mean", "gst_mean", "prs_mean", "rhu", "win_mean"]

    for days in (7, 14):
        rolling_sum = met[sum_cols].rolling(f"{days}D", min_periods=1).sum()
        rolling_mean = met[mean_cols].rolling(f"{days}D", min_periods=1).mean()
        for col in sum_cols:
            base[f"{col}_sum_{days}d"] = rolling_sum[col].reindex(base.index)
        for col in mean_cols:
            base[f"{col}_mean_{days}d"] = rolling_mean[col].reindex(base.index)

    result = flow.set_index("date")[["q"]].join(base)
    result = result.reset_index()
    return result.replace([np.inf, -np.inf], np.nan).dropna().copy()


def validate() -> None:
    """Validate source hashes and write basin summary and model-ready CSVs."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    zip_path = data_dir / args.zip_name
    metadata_path = data_dir / "zenodo_5729444_metadata.json"

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    hash_rows = []
    for filename, expected in EXPECTED_MD5.items():
        path = data_dir / filename
        if not path.exists():
            hash_rows.append(
                {
                    "file": filename,
                    "exists": False,
                    "md5": "",
                    "expected_md5": expected,
                    "md5_ok": False,
                }
            )
            continue
        actual = md5sum(path)
        hash_rows.append(
            {
                "file": filename,
                "exists": True,
                "md5": actual,
                "expected_md5": expected,
                "md5_ok": actual.lower() == expected.lower(),
            }
        )

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with ZipFile(zip_path) as z:
        names = set(z.namelist())
        natural_text = read_text_from_zip(z, "7_HydroMLYR/natural_basins.txt")
        natural_basins = {
            line.strip() for line in natural_text.splitlines() if line.strip()
        }

        rows = []
        model_frames = []
        for basin in basin_ids(z):
            met_path = f"7_HydroMLYR/1_data/{basin}/meteorological.txt"
            raw_path = f"7_HydroMLYR/1_data/{basin}/streamflow_raw.txt"
            cont_path = f"7_HydroMLYR/1_data/{basin}/continuous.txt"

            met = read_csv_from_zip(z, met_path)
            raw = read_csv_from_zip(z, raw_path)
            cont = read_csv_from_zip(z, cont_path) if cont_path in names else None

            met_cols_ok = list(met.columns) == MET_COLUMNS
            raw_cols_ok = list(raw.columns) == ["date", "q"]
            cont_cols_ok = cont is not None and list(cont.columns) == ["date", "q"]

            raw_date_match = int(raw["date"].isin(set(met["date"])).sum())
            cont_date_match = (
                int(cont["date"].isin(set(met["date"])).sum())
                if cont is not None
                else 0
            )

            row = {
                "basin_id": basin,
                "is_natural": basin in natural_basins,
                "meteorological_rows": len(met),
                "meteorological_start": met["date"].min().date(),
                "meteorological_end": met["date"].max().date(),
                "meteorological_columns_ok": met_cols_ok,
                "streamflow_raw_rows": len(raw),
                "streamflow_raw_start": raw["date"].min().date(),
                "streamflow_raw_end": raw["date"].max().date(),
                "streamflow_raw_columns_ok": raw_cols_ok,
                "raw_dates_matched_meteorology": raw_date_match,
                "has_continuous": cont is not None,
                "continuous_rows": 0 if cont is None else len(cont),
                "continuous_start": "" if cont is None else cont["date"].min().date(),
                "continuous_end": "" if cont is None else cont["date"].max().date(),
                "continuous_columns_ok": cont_cols_ok,
                "continuous_dates_matched_meteorology": cont_date_match,
                "q_raw_min": raw["q"].min(),
                "q_raw_max": raw["q"].max(),
                "q_raw_missing": int(raw["q"].isna().sum()),
            }
            rows.append(row)

            if cont is not None and not cont.empty:
                model_df = add_window_features(met, cont)
                if not model_df.empty:
                    model_df.insert(0, "region_id", basin)
                    model_df.insert(2, "is_natural", basin in natural_basins)
                    model_frames.append(model_df)

    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / args.summary_csv, index=False, encoding="utf-8-sig")

    if model_frames:
        model_ready = pd.concat(model_frames, ignore_index=True)
        model_ready["date"] = model_ready["date"].dt.strftime("%Y-%m-%d")
        model_ready.to_csv(data_dir / args.model_csv, index=False, encoding="utf-8-sig")
    else:
        model_ready = pd.DataFrame()

    print("source_title=", metadata.get("metadata", {}).get("title", ""))
    print("source_doi=", metadata.get("doi", ""))
    print("hash_validation=")
    print(pd.DataFrame(hash_rows).to_string(index=False))
    print("basins=", len(summary))
    print("natural_basins=", int(summary["is_natural"].sum()))
    print("basins_with_continuous=", int(summary["has_continuous"].sum()))
    print("met_rows_min=", int(summary["meteorological_rows"].min()))
    print("met_rows_max=", int(summary["meteorological_rows"].max()))
    print("raw_streamflow_rows_total=", int(summary["streamflow_raw_rows"].sum()))
    print("continuous_rows_total=", int(summary["continuous_rows"].sum()))
    print("all_met_columns_ok=", bool(summary["meteorological_columns_ok"].all()))
    print("all_raw_columns_ok=", bool(summary["streamflow_raw_columns_ok"].all()))
    print(
        "all_existing_continuous_columns_ok=",
        bool(summary.loc[summary["has_continuous"], "continuous_columns_ok"].all()),
    )
    print("model_ready_rows=", len(model_ready))
    print("model_ready_basins=", model_ready["region_id"].nunique() if len(model_ready) else 0)
    print("summary_csv=", data_dir / args.summary_csv)
    print("model_csv=", data_dir / args.model_csv)


if __name__ == "__main__":
    validate()
