"""
exploratory data analysis module for nfl wide receiver rookie prediction
creates visualizations and statistical summaries
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# project paths and imports (portable)
# ------------------------------------------------------------------------------

# base directory is the folder containing this file
BASE_DIR = Path(__file__).resolve().parent

# outputs and figures live under the project directory
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# allow local project imports
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

# prefer "src/utils.py"; fall back to "utils.py" at project root
try:
    from src.utils import setup_plotting_style, save_figure, print_data_summary
except ModuleNotFoundError:
    from utils import setup_plotting_style, save_figure, print_data_summary

# ------------------------------------------------------------------------------
# io helpers
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

def load_cleaned_data() -> pd.DataFrame:
    """load cleaned dataset from outputs directory"""
    parquet_path = OUTPUT_DIR / "cleaned_dataset.parquet"
    csv_path = OUTPUT_DIR / "cleaned_dataset.csv"

    # try parquet first (if engine available and file exists)
    if _parquet_available() and parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            print(f"loaded cleaned dataset (parquet): {df.shape}")
            return df
        except Exception as e:
            print(f"parquet load failed: {e}")

    # fallback to csv
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"loaded cleaned dataset (csv): {df.shape}")
            return df
        except Exception as e:
            print(f"csv load failed: {e}")

    print("cleaned dataset not found in outputs/")
    return pd.DataFrame()

# ------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------

def create_target_distribution_plot(df: pd.DataFrame):
    """create visualization of target variable distribution"""
    print("creating target variable distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if "has_1000_yard_season" in df.columns:
        target_counts = df["has_1000_yard_season"].value_counts()
        target_pct = df["has_1000_yard_season"].value_counts(normalize=True) * 100

        axes[0].bar(
            ["no 1000+ seasons", "has 1000+ seasons"],
            target_counts.values,
            color=["lightcoral", "lightblue"],
            alpha=0.7,
        )
        axes[0].set_title("distribution of 1000+ yard season achievement")
        axes[0].set_ylabel("count")

        for i, (count, pct) in enumerate(zip(target_counts.values, target_pct.values)):
            axes[0].text(
                i,
                count + 5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    if "thousand_yard_seasons" in df.columns and df["thousand_yard_seasons"].notna().any():
        thousand_yard_data = df["thousand_yard_seasons"].dropna()
        axes[1].hist(
            thousand_yard_data,
            bins=range(int(thousand_yard_data.max()) + 2),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[1].set_title("distribution of total 1000+ yard seasons")
        axes[1].set_xlabel("number of 1000+ yard seasons")
        axes[1].set_ylabel("count")
        axes[1].set_xticks(range(int(thousand_yard_data.max()) + 1))

    plt.tight_layout()
    # save via utility, targeting figs directory
    save_figure(fig, str(FIG_DIR / "target_distribution.png"))

def create_draft_analysis_plots(df: pd.DataFrame):
    """create visualizations analyzing draft position and success"""
    print("creating draft position analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    if "draft_round" in df.columns:
        draft_round_counts = df["draft_round"].value_counts().sort_index()
        axes[0, 0].bar(
            draft_round_counts.index, draft_round_counts.values, color="lightgreen", alpha=0.7
        )
        axes[0, 0].set_title("distribution by draft round")
        axes[0, 0].set_xlabel("draft round")
        axes[0, 0].set_ylabel("count")
        try:
            axes[0, 0].set_xticks(sorted([int(x) for x in draft_round_counts.index if x == x]))
        except Exception:
            pass

    if "draft_pick" in df.columns:
        draft_picks = df["draft_pick"][df["draft_pick"] > 0]
        axes[0, 1].hist(draft_picks, bins=30, alpha=0.7, color="orange", edgecolor="black")
        axes[0, 1].set_title("distribution of draft pick numbers")
        axes[0, 1].set_xlabel("draft pick")
        axes[0, 1].set_ylabel("count")

    if {"draft_round", "has_1000_yard_season"}.issubset(df.columns):
        success_by_round = df.groupby("draft_round")["has_1000_yard_season"].agg(["mean", "count"])
        success_by_round = success_by_round[success_by_round["count"] >= 5]
        axes[1, 0].bar(success_by_round.index, success_by_round["mean"] * 100, color="purple", alpha=0.7)
        axes[1, 0].set_title("success rate by draft round")
        axes[1, 0].set_xlabel("draft round")
        axes[1, 0].set_ylabel("success rate (%)")
        for idx, row in success_by_round.iterrows():
            axes[1, 0].text(idx, row["mean"] * 100 + 1, f"n={row['count']}", ha="center", va="bottom", fontsize=8)

    if {"draft_pick", "has_1000_yard_season"}.issubset(df.columns):
        draft_success_df = df[(df["draft_pick"] > 0) & (df["draft_pick"] <= 250)].copy()
        if not draft_success_df.empty:
            draft_success_df["draft_pick_bin"] = pd.cut(draft_success_df["draft_pick"], bins=10, labels=False)
            bin_success = draft_success_df.groupby("draft_pick_bin").agg(
                {"has_1000_yard_season": "mean", "draft_pick": "mean"}
            )
            axes[1, 1].scatter(
                bin_success["draft_pick"], bin_success["has_1000_yard_season"] * 100, s=100, alpha=0.7, color="red"
            )
            axes[1, 1].set_title("success rate vs draft position")
            axes[1, 1].set_xlabel("average draft pick")
            axes[1, 1].set_ylabel("success rate (%)")
            z = np.polyfit(bin_success["draft_pick"], bin_success["has_1000_yard_season"] * 100, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(bin_success["draft_pick"], p(bin_success["draft_pick"]), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    save_figure(fig, str(FIG_DIR / "draft_analysis.png"))

def create_rookie_performance_plots(df: pd.DataFrame):
    """create visualizations of rookie season performance metrics"""
    print("creating rookie performance analysis plots...")

    performance_cols = ["rec", "rec_yards", "rec_td", "targets", "catch_rate", "yards_per_reception"]
    available_cols = [c for c in performance_cols if c in df.columns]

    if len(available_cols) < 2:
        print("insufficient performance columns for analysis")
        return

    n_cols = min(3, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(available_cols[: len(axes)]):
        if "has_1000_yard_season" in df.columns:
            success_data = df[df["has_1000_yard_season"] == 1][col].dropna()
            no_success_data = df[df["has_1000_yard_season"] == 0][col].dropna()

            axes[i].hist(no_success_data, bins=30, alpha=0.6, label="no 1000+ seasons", color="lightcoral", density=True)
            axes[i].hist(success_data, bins=30, alpha=0.6, label="has 1000+ seasons", color="lightblue", density=True)

            axes[i].set_title(f"distribution of {col.replace('_', ' ')}")
            axes[i].set_xlabel(col.replace("_", " "))
            axes[i].set_ylabel("density")
            axes[i].legend()

            if len(success_data) > 0:
                axes[i].axvline(success_data.mean(), color="blue", linestyle="--", alpha=0.8)
            if len(no_success_data) > 0:
                axes[i].axvline(no_success_data.mean(), color="red", linestyle="--", alpha=0.8)

    for j in range(len(available_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_figure(fig, str(FIG_DIR / "rookie_performance_distributions.png"))

def create_correlation_heatmap(df: pd.DataFrame):
    """create correlation heatmap of key numeric variables"""
    print("creating correlation heatmap...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_vars = [
        "draft_round",
        "draft_pick",
        "age",
        "rec",
        "rec_yards",
        "rec_td",
        "targets",
        "catch_rate",
        "yards_per_reception",
        "yards_per_route_run",
        "has_1000_yard_season",
        "thousand_yard_seasons",
    ]
    cols = [c for c in key_vars if c in numeric_cols]

    if len(cols) < 3:
        print("insufficient numeric columns for correlation analysis")
        return

    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="RdBu_r", center=0, square=True, fmt=".2f", cbar_kws={"shrink": 0.8})
    ax.set_title("correlation matrix of key variables")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_figure(fig, str(FIG_DIR / "correlation_heatmap.png"))

def create_time_trend_analysis(df: pd.DataFrame):
    """analyze trends over time based on rookie year"""
    print("creating time trend analysis...")

    if "rookie_year" not in df.columns:
        print("rookie_year column not available for time trend analysis")
        return

    year_df = df[(df["rookie_year"] >= 2006) & (df["rookie_year"] <= 2024)].copy()
    if year_df.empty:
        print("no data in reasonable year range")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    if "has_1000_yard_season" in year_df.columns:
        yearly_success = year_df.groupby("rookie_year").agg({"has_1000_yard_season": ["mean", "count"]}).round(3)
        yearly_success.columns = ["success_rate", "count"]
        yearly_success = yearly_success[yearly_success["count"] >= 3]
        axes[0, 0].plot(yearly_success.index, yearly_success["success_rate"] * 100, marker="o", linewidth=2, markersize=6)
        axes[0, 0].set_title("success rate by rookie year")
        axes[0, 0].set_xlabel("rookie year")
        axes[0, 0].set_ylabel("success rate (%)")
        axes[0, 0].grid(True, alpha=0.3)

    yearly_counts = year_df["rookie_year"].value_counts().sort_index()
    axes[0, 1].bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color="green")
    axes[0, 1].set_title("number of wr rookies by year")
    axes[0, 1].set_xlabel("rookie year")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].tick_params(axis="x", rotation=45)

    performance_metrics = ["rec_yards", "rec", "targets"]
    available_metrics = [m for m in performance_metrics if m in year_df.columns]
    if available_metrics:
        yearly_perf = year_df.groupby("rookie_year")[available_metrics].mean()
        for i, metric in enumerate(available_metrics[:2]):
            ax = axes[1, i]
            ax.plot(yearly_perf.index, yearly_perf[metric], marker="s", linewidth=2, markersize=6)
            ax.set_title(f"average {metric.replace('_', ' ')} by rookie year")
            ax.set_xlabel("rookie year")
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

    if len(available_metrics) < 2:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    save_figure(fig, str(FIG_DIR / "time_trends.png"))

def create_outlier_analysis(df: pd.DataFrame):
    """identify and visualize outliers in key performance metrics"""
    print("creating outlier analysis...")

    performance_cols = ["rec", "rec_yards", "rec_td", "targets", "yards_per_reception"]
    available_cols = [c for c in performance_cols if c in df.columns]

    if len(available_cols) < 2:
        print("insufficient columns for outlier analysis")
        return

    n_cols = min(3, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
    axes = axes.flatten()

    outlier_summary = []

    for i, col in enumerate(available_cols[: len(axes)]):
        data = df[col].dropna()
        if len(data) == 0:
            continue

        if "has_1000_yard_season" in df.columns:
            success_data = df[df["has_1000_yard_season"] == 1][col].dropna()
            no_success_data = df[df["has_1000_yard_season"] == 0][col].dropna()

            bp = axes[i].boxplot([no_success_data, success_data], labels=["no 1000+ seasons", "has 1000+ seasons"], patch_artist=True)
            bp["boxes"][0].set_facecolor("lightcoral")
            bp["boxes"][1].set_facecolor("lightblue")
        else:
            axes[i].boxplot(data)

        axes[i].set_title(f"outlier analysis: {col.replace('_', ' ')}")
        axes[i].set_ylabel(col.replace("_", " "))
        axes[i].tick_params(axis="x", rotation=45)

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outs = data[(data < lower) | (data > upper)]
        outlier_summary.append(
            {
                "metric": col,
                "outlier_count": int(len(outs)),
                "outlier_percentage": float(len(outs) / len(data) * 100),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
        )

    for j in range(len(available_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_figure(fig, str(FIG_DIR / "outlier_analysis.png"))

    # save outlier summary under outputs
    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)
    print("outlier summary saved to outputs/outlier_summary.csv")

def create_feature_importance_preview(df: pd.DataFrame):
    """create preliminary feature importance analysis using correlation with target"""
    print("creating feature importance preview...")

    if "has_1000_yard_season" not in df.columns:
        print("target variable not available for feature importance analysis")
        return

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "has_1000_yard_season"]

    correlations = []
    for col in numeric_cols:
        if df[col].notna().sum() > 10:
            corr = df[col].corr(df["has_1000_yard_season"])
            if not np.isnan(corr):
                correlations.append({"feature": col, "correlation": float(corr), "abs_correlation": float(abs(corr))})

    if not correlations:
        print("no valid correlations found")
        return

    corr_df = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False)
    top_features = corr_df.head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["red" if x < 0 else "blue" for x in top_features["correlation"]]
    bars = ax.barh(range(len(top_features)), top_features["correlation"], color=colors, alpha=0.7)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].str.replace("_", " ").str.lower())
    ax.set_xlabel("correlation with 1000+ yard season success")
    ax.set_title("top 15 features by correlation with target")
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="x")

    for i, (bar, corr) in enumerate(zip(bars, top_features["correlation"])):
        ax.text(
            corr + (0.01 if corr >= 0 else -0.01),
            i,
            f"{corr:.3f}",
            va="center",
            ha="left" if corr >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    save_figure(fig, str(FIG_DIR / "feature_importance_preview.png"))

    top_features.to_csv(OUTPUT_DIR / "feature_correlations.csv", index=False)
    print("feature correlations saved to outputs/feature_correlations.csv")

# ------------------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------------------

def generate_eda_summary_report(df: pd.DataFrame):
    """generate eda summary report"""
    print("generating eda summary report...")

    lines = []
    lines.append("# exploratory data analysis summary report\n\n")

    lines.append("## dataset overview\n")
    lines.append(f"- **total records:** {len(df):,}\n")
    lines.append(f"- **total features:** {len(df.columns)}\n")
    lines.append(f"- **numeric features:** {len(df.select_dtypes(include=[np.number]).columns)}\n")
    lines.append(f"- **categorical features:** {len(df.select_dtypes(include=['object']).columns)}\n")

    if "has_1000_yard_season" in df.columns:
        vc = df["has_1000_yard_season"].value_counts()
        success_rate = vc.get(1, 0) / max(len(df), 1) * 100
        lines.append("\n## target variable analysis\n")
        lines.append(f"- **overall success rate:** {success_rate:.1f}%\n")
        lines.append(f"- **successful players:** {vc.get(1, 0):,}\n")
        lines.append(f"- **unsuccessful players:** {vc.get(0, 0):,}\n")
        imbalance_ratio = vc.get(0, 0) / max(vc.get(1, 1), 1)
        lines.append(f"- **class imbalance ratio:** {imbalance_ratio:.1f}:1\n")

    if {"draft_round", "has_1000_yard_season"}.issubset(df.columns):
        draft_analysis = df.groupby("draft_round").agg({"has_1000_yard_season": ["count", "sum", "mean"]}).round(3)
        draft_analysis.columns = ["total", "successful", "success_rate"]
        lines.append("\n## draft position analysis\n")
        lines.append("| round | total | successful | success rate |\n")
        lines.append("|-------|-------|------------|--------------|\n")
        for round_num, row in draft_analysis.iterrows():
            if row["total"] >= 5:
                lines.append(f"| {round_num} | {int(row['total'])} | {int(row['successful'])} | {row['success_rate']*100:.1f}% |\n")

    performance_cols = ["rec", "rec_yards", "rec_td", "targets", "catch_rate"]
    available_perf = [c for c in performance_cols if c in df.columns]
    if available_perf and "has_1000_yard_season" in df.columns:
        lines.append("\n## rookie performance comparison\n")
        lines.append("| metric | successful avg | unsuccessful avg | difference |\n")
        lines.append("|--------|----------------|------------------|------------|\n")
        for col in available_perf:
            s_avg = df[df["has_1000_yard_season"] == 1][col].mean()
            n_avg = df[df["has_1000_yard_season"] == 0][col].mean()
            if not (np.isnan(s_avg) or np.isnan(n_avg)):
                diff = s_avg - n_avg
                lines.append(f"| {col.replace('_', ' ')} | {s_avg:.2f} | {n_avg:.2f} | {diff:.2f} |\n")

    missing = df.isnull().sum()
    missing_pct = (missing / max(len(df), 1)) * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    if len(high_missing) > 0:
        lines.append("\n## high missing data features (>50%)\n")
        for col, pct in high_missing.head(10).items():
            lines.append(f"- **{col}:** {pct:.1f}% missing\n")

    lines.append("\n## key insights\n")
    if {"draft_round", "has_1000_yard_season"}.issubset(df.columns):
        early = df[df["draft_round"] <= 2]["has_1000_yard_season"].mean() * 100
        late = df[df["draft_round"] >= 5]["has_1000_yard_season"].mean() * 100
        if not np.isnan(early):
            lines.append(f"- early round picks (1-2) have {early:.1f}% success rate\n")
        if not np.isnan(late):
            lines.append(f"- late round picks (5+) have {late:.1f}% success rate\n")

    if {"rec_yards", "has_1000_yard_season"}.issubset(df.columns):
        high = df[df["rec_yards"] >= 500]["has_1000_yard_season"].mean() * 100
        low = df[df["rec_yards"] < 200]["has_1000_yard_season"].mean() * 100
        if not np.isnan(high):
            lines.append(f"- rookies with 500+ yards have {high:.1f}% success rate\n")
        if not np.isnan(low):
            lines.append(f"- rookies with <200 yards have {low:.1f}% success rate\n")

    report_path = OUTPUT_DIR / "eda_summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"eda summary report saved to {report_path}")

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():
    """run exploratory data analysis"""
    print("starting exploratory data analysis")
    print("=" * 50)

    df = load_cleaned_data()
    if df.empty:
        print("no data available for analysis")
        return

    setup_plotting_style()

    create_target_distribution_plot(df)
    create_draft_analysis_plots(df)
    create_rookie_performance_plots(df)
    create_correlation_heatmap(df)
    create_time_trend_analysis(df)
    create_outlier_analysis(df)
    create_feature_importance_preview(df)

    generate_eda_summary_report(df)

    print("\nexploratory data analysis completed successfully")
    print(f"figures saved to {FIG_DIR}")
    print(f"reports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

