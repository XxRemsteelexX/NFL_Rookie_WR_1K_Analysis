"""
data integration and cleaning module for nfl wide receiver rookie prediction
combines raw csv files, handles missing values, and creates unified dataset
"""

from __future__ import annotations

import os
import sys
import glob
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# project paths and imports
# ------------------------------------------------------------------------------

# base directory is the folder containing this file
BASE_DIR = Path(__file__).resolve().parent

# outputs and figures live under the project directory
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# data directory can be overridden via env var; default is "Uploads" under project
DATA_DIR = Path(os.environ.get("WR_DATA_DIR", BASE_DIR / "Uploads"))

# allow local project imports
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

# prefer "src/utils.py"; fall back to "utils.py" at project root
try:
    from src.utils import (
        standardize_player_name,
        standardize_team_name,
        get_column_mapping,
        clean_numeric_column,
        print_data_summary,
    )
except ModuleNotFoundError:
    from utils import (
        standardize_player_name,
        standardize_team_name,
        get_column_mapping,
        clean_numeric_column,
        print_data_summary,
    )

# ------------------------------------------------------------------------------
# loader helpers
# ------------------------------------------------------------------------------

def _read_csv_with_fallbacks(path: str | Path) -> pd.DataFrame:
    """read csv trying multiple encodings"""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8/latin-1/cp1252", b"", 0, 1, f"unable to decode {path}")

def load_receiving_data() -> pd.DataFrame:
    """load and combine receiving statistics files (rec20xx.csv) from data dir"""
    print("loading receiving statistics data...")
    pattern = str(DATA_DIR / "rec20*.csv")
    rec_files = sorted(glob.glob(pattern))
    if not rec_files:
        print(f"  no files matched: {pattern}")
        return pd.DataFrame()

    frames = []
    for file in rec_files:
        try:
            stem = Path(file).stem  # e.g., rec2021
            year = int(stem.replace("rec", ""))
            print(f"  processing {file} (year: {year})")

            df = _read_csv_with_fallbacks(file)
            df["season"] = year

            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)

            if "player_name" in df.columns:
                df["player_name"] = df["player_name"].apply(standardize_player_name)
            if "team" in df.columns:
                df["team"] = df["team"].apply(standardize_team_name)

            frames.append(df)
        except Exception as e:
            print(f"  error loading {file}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"combined receiving data: {out.shape}")
    return out

def load_rookie_data() -> pd.DataFrame:
    """load and combine rookie draft files"""
    print("loading rookie draft data...")
    files = []
    files.extend(glob.glob(str(DATA_DIR / "*rookie*.csv")))
    files.extend(glob.glob(str(DATA_DIR / "20*rookies.csv")))
    files = sorted(set(files))
    if not files:
        print("  no rookie files found")
        return pd.DataFrame()

    frames = []
    for file in files:
        try:
            df = _read_csv_with_fallbacks(file)

            # infer year from filename parts
            rookie_year = None
            for part in Path(file).parts:
                digits = "".join(ch for ch in part if ch.isdigit())
                if len(digits) == 4 and digits.startswith("20"):
                    rookie_year = int(digits)
                    break
            if rookie_year is not None:
                df["rookie_year"] = rookie_year

            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)

            # filter for wide receivers
            if "Pos" in df.columns:
                df = df[df["Pos"] == "WR"].copy()
            elif "position" in df.columns:
                df = df[df["position"] == "WR"].copy()

            if "player_name" in df.columns:
                df["player_name"] = df["player_name"].apply(standardize_player_name)
            if "team" in df.columns:
                df["team"] = df["team"].apply(standardize_team_name)

            frames.append(df)
        except Exception as e:
            print(f"  error loading {file}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    # drop duplicates by player and rookie year if available
    if {"player_name", "rookie_year"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["player_name", "rookie_year"], keep="first")

    print(f"combined rookie data: {out.shape}")
    return out

def load_advanced_metrics() -> pd.DataFrame:
    """load and combine advanced wr metrics"""
    print("loading advanced metrics data...")
    files = sorted(glob.glob(str(DATA_DIR / "nfl-advanced-wr-*.csv")))
    if not files:
        print("  no advanced metric files found")
        return pd.DataFrame()

    frames = []
    for file in files:
        try:
            df = _read_csv_with_fallbacks(file)

            # infer season from filename parts split by '-'
            season = None
            for part in Path(file).stem.split("-"):
                if part.isdigit() and len(part) == 4:
                    season = int(part)
                    break
            if season is not None:
                df["season"] = season

            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)

            # create standardized player_name if only "player" exists
            if "player" in df.columns and "player_name" not in df.columns:
                df["player_name"] = df["player"].apply(standardize_player_name)
            elif "player_name" in df.columns:
                df["player_name"] = df["player_name"].apply(standardize_player_name)

            if "team" in df.columns:
                df["team"] = df["team"].apply(standardize_team_name)

            frames.append(df)
        except Exception as e:
            print(f"  error loading {file}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    print(f"combined advanced metrics: {out.shape}")
    return out

def load_target_data() -> pd.DataFrame:
    """load target labels for 1000+ yard seasons"""
    print("loading target variable data...")
    path = DATA_DIR / "1kseasons.csv"
    if not path.exists():
        print(f"  missing target file: {path}")
        return pd.DataFrame()

    try:
        df = _read_csv_with_fallbacks(path)
        df = df.rename(
            columns={
                "Player": "player_name",
                "Rookie year": "rookie_year",
                "1,000 Yard Seasons": "thousand_yard_seasons",
            }
        )

        df["player_name"] = df["player_name"].apply(standardize_player_name)
        df["rookie_year"] = clean_numeric_column(df["rookie_year"])
        df["thousand_yard_seasons"] = clean_numeric_column(df["thousand_yard_seasons"])
        df["has_1000_yard_season"] = (df["thousand_yard_seasons"] > 0).astype(int)

        print(f"target data loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"  error loading target data: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# merge and clean
# ------------------------------------------------------------------------------

def merge_all_datasets(
    rec_data: pd.DataFrame,
    rookie_data: pd.DataFrame,
    advanced_data: pd.DataFrame,
    target_data: pd.DataFrame,
) -> pd.DataFrame:
    """merge rookie base with receiving, advanced metrics, and target labels"""
    print("merging all datasets...")

    merged = rookie_data.copy()
    print(f"starting with rookie data: {merged.shape}")

    # receiving merge on player + rookie year
    if not rec_data.empty and {"player_name", "season"}.issubset(rec_data.columns) and {
        "player_name",
        "rookie_year",
    }.issubset(merged.columns):
        rec_copy = rec_data.copy()
        rec_copy["merge_key"] = rec_copy["player_name"] + "_" + rec_copy["season"].astype(str)

        merged["merge_key"] = merged["player_name"] + "_" + merged["rookie_year"].astype(str)
        merged = pd.merge(
            merged,
            rec_copy,
            on="merge_key",
            how="left",
            suffixes=("", "_rec"),
        ).drop(columns=["merge_key"])
        print(f"after merging receiving data: {merged.shape}")

    # advanced merge on player + rookie year
    if not advanced_data.empty and {"player_name", "season"}.issubset(advanced_data.columns) and {
        "player_name",
        "rookie_year",
    }.issubset(merged.columns):
        adv_copy = advanced_data.copy()
        adv_copy["merge_key"] = adv_copy["player_name"] + "_" + adv_copy["season"].astype(str)

        merged["merge_key"] = merged["player_name"] + "_" + merged["rookie_year"].astype(str)
        merged = pd.merge(
            merged,
            adv_copy,
            on="merge_key",
            how="left",
            suffixes=("", "_adv"),
        ).drop(columns=["merge_key"])
        print(f"after merging advanced metrics: {merged.shape}")
    else:
        print("advanced metrics missing required columns for merge or empty")

    # target merge
    if not target_data.empty and {"player_name", "rookie_year"}.issubset(merged.columns) and {
        "player_name",
        "rookie_year",
    }.issubset(target_data.columns):
        cols = ["player_name", "rookie_year", "has_1000_yard_season", "thousand_yard_seasons"]
        cols = [c for c in cols if c in target_data.columns]
        merged = pd.merge(merged, target_data[cols], on=["player_name", "rookie_year"], how="left")
        print(f"after merging target data: {merged.shape}")

    return merged

def clean_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """clean merged dataset and add simple derived features"""
    print("cleaning merged dataset...")
    out = df.copy()

    # drop rows without player_name
    if "player_name" in out.columns:
        out = out.dropna(subset=["player_name"])

    # numeric cleaning
    numeric_cols = [
        "draft_round",
        "draft_pick",
        "age",
        "rec",
        "rec_yards",
        "rec_td",
        "targets",
        "routes_run",
        "air_yards",
        "yac",
        "target_share",
        "route_participation",
        "contested_catch_pct",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = clean_numeric_column(out[col])

    # target fill
    if "has_1000_yard_season" not in out.columns:
        out["has_1000_yard_season"] = 0
    else:
        out["has_1000_yard_season"] = out["has_1000_yard_season"].fillna(0)

    # derived features
    if {"rec", "targets"}.issubset(out.columns):
        out["catch_rate"] = np.where(out["targets"] > 0, out["rec"] / out["targets"], 0)

    if {"rec_yards", "rec"}.issubset(out.columns):
        out["yards_per_reception"] = np.where(out["rec"] > 0, out["rec_yards"] / out["rec"], 0)

    if {"rec_yards", "routes_run"}.issubset(out.columns):
        out["yards_per_route_run"] = np.where(out["routes_run"] > 0, out["rec_yards"] / out["routes_run"], 0)

    # unique by player and rookie year if present
    if {"player_name", "rookie_year"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["player_name", "rookie_year"], keep="first")

    print(f"cleaned dataset shape: {out.shape}")
    return out

# ------------------------------------------------------------------------------
# save helpers
# ------------------------------------------------------------------------------

def _parquet_available() -> bool:
    """check whether a parquet engine is available"""
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except Exception:
        return False

def save_clean_outputs(df: pd.DataFrame) -> None:
    """save cleaned dataset to parquet if possible, else csv"""
    parquet_path = OUTPUT_DIR / "cleaned_dataset.parquet"
    csv_path = OUTPUT_DIR / "cleaned_dataset.csv"

    if _parquet_available():
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"\ncleaned dataset saved to: {parquet_path}")
        except Exception as e:
            print(f"parquet save failed: {e}")
            df.to_csv(csv_path, index=False)
            print(f"csv fallback saved to: {csv_path}")
            print("install 'pyarrow' or 'fastparquet' to enable parquet saves")
    else:
        df.to_csv(csv_path, index=False)
        print(f"\nparquet engine not found; csv saved to: {csv_path}")
        print("install 'pyarrow' or 'fastparquet' to enable parquet saves")

def create_data_profile(df: pd.DataFrame) -> None:
    """write a lightweight markdown data profile"""
    print("creating data profile report...")

    lines = []
    lines.append("# nfl wr rookie prediction - data profile report\n")
    lines.append(f"**dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns\n")
    lines.append(f"**memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} mb\n")

    if "has_1000_yard_season" in df.columns:
        vc = df["has_1000_yard_season"].value_counts()
        n0 = int(vc.get(0, 0))
        n1 = int(vc.get(1, 0))
        tot = max(len(df), 1)
        lines.append("\n## target variable distribution\n")
        lines.append(f"- no 1000+ yard seasons: {n0} ({n0 / tot * 100:.1f}%)\n")
        lines.append(f"- has 1000+ yard seasons: {n1} ({n1 / tot * 100:.1f}%)\n")

    missing = df.isnull().sum()
    missing_pct = (missing / max(len(df), 1)) * 100
    miss_df = (
        pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
        .sort_values("missing_count", ascending=False)
    )

    lines.append("\n## missing values summary\n")
    for col, row in miss_df[miss_df["missing_count"] > 0].iterrows():
        lines.append(f"- {col}: {int(row['missing_count'])} ({row['missing_percent']:.1f}%)\n")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        lines.append("\n## numeric columns summary\n")
        desc = df[num_cols].describe()
        lines.append(desc.to_string())
        lines.append("\n")

    profile_path = OUTPUT_DIR / "data_profile_report.md"
    with open(profile_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"data profile report saved to: {profile_path}")

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main() -> None:
    """run data integration and cleaning pipeline"""
    print("starting nfl wr rookie prediction data integration pipeline")
    print("=" * 60)

    rec = load_receiving_data()
    rookies = load_rookie_data()
    adv = load_advanced_metrics()
    target = load_target_data()

    print("\ndebugging column names:")
    print(f"rookie data columns: {list(rookies.columns) if not rookies.empty else 'empty'}")
    print(f"advanced data columns: {list(adv.columns) if not adv.empty else 'empty'}")

    merged = merge_all_datasets(rec, rookies, adv, target)
    clean = clean_merged_dataset(merged)

    print_data_summary(clean, "final cleaned dataset")

    save_clean_outputs(clean)
    create_data_profile(clean)

    print("\ndata integration pipeline completed successfully!")

if __name__ == "__main__":
    main()
