
"""
exploratory data analysis module for nfl wide receiver rookie prediction
creates comprehensive visualizations and statistical summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils import setup_plotting_style, save_figure, print_data_summary

def load_cleaned_data() -> pd.DataFrame:
    """load the cleaned dataset from parquet file"""
    try:
        df = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/cleaned_dataset.parquet')
        print(f"loaded cleaned dataset: {df.shape}")
        return df
    except Exception as e:
        print(f"error loading cleaned dataset: {e}")
        return pd.DataFrame()

def create_target_distribution_plot(df: pd.DataFrame):
    """create visualization of target variable distribution"""
    print("creating target variable distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # binary target distribution
    if 'has_1000_yard_season' in df.columns:
        target_counts = df['has_1000_yard_season'].value_counts()
        target_pct = df['has_1000_yard_season'].value_counts(normalize=True) * 100
        
        axes[0].bar(['No 1000+ Seasons', 'Has 1000+ Seasons'], target_counts.values, 
                   color=['lightcoral', 'lightblue'], alpha=0.7)
        axes[0].set_title('Distribution of 1000+ Yard Season Achievement')
        axes[0].set_ylabel('Count')
        
        # add percentage labels
        for i, (count, pct) in enumerate(zip(target_counts.values, target_pct.values)):
            axes[0].text(i, count + 5, f'{count}\n({pct:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
    
    # continuous target distribution
    if 'thousand_yard_seasons' in df.columns:
        thousand_yard_data = df['thousand_yard_seasons'].dropna()
        axes[1].hist(thousand_yard_data, bins=range(int(thousand_yard_data.max()) + 2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_title('Distribution of Total 1000+ Yard Seasons')
        axes[1].set_xlabel('Number of 1000+ Yard Seasons')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(int(thousand_yard_data.max()) + 1))
    
    plt.tight_layout()
    save_figure(fig, 'target_distribution.png')

def create_draft_analysis_plots(df: pd.DataFrame):
    """create visualizations analyzing draft position and success"""
    print("creating draft position analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # draft round distribution
    if 'draft_round' in df.columns:
        draft_round_counts = df['draft_round'].value_counts().sort_index()
        axes[0, 0].bar(draft_round_counts.index, draft_round_counts.values, 
                      color='lightgreen', alpha=0.7)
        axes[0, 0].set_title('Distribution by Draft Round')
        axes[0, 0].set_xlabel('Draft Round')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(1, 8))
    
    # draft pick distribution
    if 'draft_pick' in df.columns:
        draft_picks = df['draft_pick'][df['draft_pick'] > 0]  # exclude undrafted
        axes[0, 1].hist(draft_picks, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('Distribution of Draft Pick Numbers')
        axes[0, 1].set_xlabel('Draft Pick')
        axes[0, 1].set_ylabel('Count')
    
    # success rate by draft round
    if 'draft_round' in df.columns and 'has_1000_yard_season' in df.columns:
        success_by_round = df.groupby('draft_round')['has_1000_yard_season'].agg(['mean', 'count'])
        success_by_round = success_by_round[success_by_round['count'] >= 5]  # minimum sample size
        
        axes[1, 0].bar(success_by_round.index, success_by_round['mean'] * 100, 
                      color='purple', alpha=0.7)
        axes[1, 0].set_title('Success Rate by Draft Round')
        axes[1, 0].set_xlabel('Draft Round')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_xticks(success_by_round.index)
        
        # add sample size labels
        for i, (idx, row) in enumerate(success_by_round.iterrows()):
            axes[1, 0].text(idx, row['mean'] * 100 + 1, f'n={row["count"]}', 
                           ha='center', va='bottom', fontsize=8)
    
    # draft pick vs success (scatter with trend)
    if 'draft_pick' in df.columns and 'has_1000_yard_season' in df.columns:
        draft_success_df = df[(df['draft_pick'] > 0) & (df['draft_pick'] <= 250)].copy()
        
        # create bins for better visualization
        draft_success_df['draft_pick_bin'] = pd.cut(draft_success_df['draft_pick'], 
                                                   bins=10, labels=False)
        bin_success = draft_success_df.groupby('draft_pick_bin').agg({
            'has_1000_yard_season': 'mean',
            'draft_pick': 'mean'
        })
        
        axes[1, 1].scatter(bin_success['draft_pick'], bin_success['has_1000_yard_season'] * 100,
                          s=100, alpha=0.7, color='red')
        axes[1, 1].set_title('Success Rate vs Draft Position')
        axes[1, 1].set_xlabel('Average Draft Pick')
        axes[1, 1].set_ylabel('Success Rate (%)')
        
        # add trend line
        z = np.polyfit(bin_success['draft_pick'], bin_success['has_1000_yard_season'] * 100, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(bin_success['draft_pick'], p(bin_success['draft_pick']), 
                       "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    save_figure(fig, 'draft_analysis.png')

def create_rookie_performance_plots(df: pd.DataFrame):
    """create visualizations of rookie season performance metrics"""
    print("creating rookie performance analysis plots...")
    
    # key performance metrics
    performance_cols = ['rec', 'rec_yards', 'rec_td', 'targets', 'catch_rate', 'yards_per_reception']
    available_cols = [col for col in performance_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("insufficient performance columns for analysis")
        return
    
    n_cols = min(3, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(available_cols):
        if i >= len(axes):
            break
            
        # distribution by success
        if 'has_1000_yard_season' in df.columns:
            success_data = df[df['has_1000_yard_season'] == 1][col].dropna()
            no_success_data = df[df['has_1000_yard_season'] == 0][col].dropna()
            
            axes[i].hist(no_success_data, bins=30, alpha=0.6, label='No 1000+ Seasons', 
                        color='lightcoral', density=True)
            axes[i].hist(success_data, bins=30, alpha=0.6, label='Has 1000+ Seasons', 
                        color='lightblue', density=True)
            
            axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
            axes[i].set_xlabel(col.replace("_", " ").title())
            axes[i].set_ylabel('Density')
            axes[i].legend()
            
            # add mean lines
            if len(success_data) > 0:
                axes[i].axvline(success_data.mean(), color='blue', linestyle='--', alpha=0.8,
                               label=f'Success Mean: {success_data.mean():.2f}')
            if len(no_success_data) > 0:
                axes[i].axvline(no_success_data.mean(), color='red', linestyle='--', alpha=0.8,
                               label=f'No Success Mean: {no_success_data.mean():.2f}')
    
    # hide empty subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'rookie_performance_distributions.png')

def create_correlation_heatmap(df: pd.DataFrame):
    """create correlation heatmap of key numeric variables"""
    print("creating correlation heatmap...")
    
    # select numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # focus on key variables
    key_vars = [
        'draft_round', 'draft_pick', 'age', 'rec', 'rec_yards', 'rec_td',
        'targets', 'catch_rate', 'yards_per_reception', 'yards_per_route_run',
        'has_1000_yard_season', 'thousand_yard_seasons'
    ]
    
    correlation_cols = [col for col in key_vars if col in numeric_cols]
    
    if len(correlation_cols) < 3:
        print("insufficient numeric columns for correlation analysis")
        return
    
    # calculate correlation matrix
    corr_matrix = df[correlation_cols].corr()
    
    # create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    ax.set_title('Correlation Matrix of Key Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_figure(fig, 'correlation_heatmap.png')

def create_time_trend_analysis(df: pd.DataFrame):
    """analyze trends over time (rookie years)"""
    print("creating time trend analysis...")
    
    if 'rookie_year' not in df.columns:
        print("rookie_year column not available for time trend analysis")
        return
    
    # filter to reasonable year range
    year_df = df[(df['rookie_year'] >= 2006) & (df['rookie_year'] <= 2024)].copy()
    
    if len(year_df) == 0:
        print("no data in reasonable year range")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # success rate by year
    if 'has_1000_yard_season' in year_df.columns:
        yearly_success = year_df.groupby('rookie_year').agg({
            'has_1000_yard_season': ['mean', 'count']
        }).round(3)
        yearly_success.columns = ['success_rate', 'count']
        yearly_success = yearly_success[yearly_success['count'] >= 3]  # minimum sample size
        
        axes[0, 0].plot(yearly_success.index, yearly_success['success_rate'] * 100, 
                       marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Success Rate by Rookie Year')
        axes[0, 0].set_xlabel('Rookie Year')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # number of rookies by year
    yearly_counts = year_df['rookie_year'].value_counts().sort_index()
    axes[0, 1].bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color='green')
    axes[0, 1].set_title('Number of WR Rookies by Year')
    axes[0, 1].set_xlabel('Rookie Year')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # average performance metrics by year
    performance_metrics = ['rec_yards', 'rec', 'targets']
    available_metrics = [col for col in performance_metrics if col in year_df.columns]
    
    if available_metrics:
        yearly_performance = year_df.groupby('rookie_year')[available_metrics].mean()
        
        for i, metric in enumerate(available_metrics[:2]):  # plot up to 2 metrics
            ax = axes[1, i]
            ax.plot(yearly_performance.index, yearly_performance[metric], 
                   marker='s', linewidth=2, markersize=6, color=f'C{i}')
            ax.set_title(f'Average {metric.replace("_", " ").title()} by Rookie Year')
            ax.set_xlabel('Rookie Year')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    # hide empty subplot if needed
    if len(available_metrics) < 2:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'time_trends.png')

def create_outlier_analysis(df: pd.DataFrame):
    """identify and visualize outliers in key performance metrics"""
    print("creating outlier analysis...")
    
    performance_cols = ['rec', 'rec_yards', 'rec_td', 'targets', 'yards_per_reception']
    available_cols = [col for col in performance_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("insufficient columns for outlier analysis")
        return
    
    n_cols = min(3, len(available_cols))
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    outlier_summary = []
    
    for i, col in enumerate(available_cols):
        if i >= len(axes):
            break
            
        # create boxplot
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # separate by success for comparison
        if 'has_1000_yard_season' in df.columns:
            success_data = df[df['has_1000_yard_season'] == 1][col].dropna()
            no_success_data = df[df['has_1000_yard_season'] == 0][col].dropna()
            
            box_data = [no_success_data, success_data]
            labels = ['No 1000+ Seasons', 'Has 1000+ Seasons']
            
            bp = axes[i].boxplot(box_data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')
        else:
            axes[i].boxplot(data)
        
        axes[i].set_title(f'Outlier Analysis: {col.replace("_", " ").title()}')
        axes[i].set_ylabel(col.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # identify outliers using iqr method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_summary.append({
            'metric': col,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(data) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    # hide empty subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'outlier_analysis.png')
    
    # save outlier summary
    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df.to_csv('/home/yeblad/Desktop/New_WR_analysis/outputs/outlier_summary.csv', index=False)
    print("outlier summary saved to outlier_summary.csv")

def create_feature_importance_preview(df: pd.DataFrame):
    """create preliminary feature importance analysis using correlation with target"""
    print("creating feature importance preview...")
    
    if 'has_1000_yard_season' not in df.columns:
        print("target variable not available for feature importance analysis")
        return
    
    # calculate correlation with target variable
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'has_1000_yard_season']
    
    correlations = []
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # minimum data points
            corr = df[col].corr(df['has_1000_yard_season'])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
    
    if not correlations:
        print("no valid correlations found")
        return
    
    # sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    top_features = corr_df.head(15)
    
    # create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if x < 0 else 'blue' for x in top_features['correlation']]
    bars = ax.barh(range(len(top_features)), top_features['correlation'], color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
    ax.set_xlabel('Correlation with 1000+ Yard Season Success')
    ax.set_title('Top 15 Features by Correlation with Target Variable')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')
    
    # add correlation values as text
    for i, (bar, corr) in enumerate(zip(bars, top_features['correlation'])):
        ax.text(corr + (0.01 if corr >= 0 else -0.01), i, f'{corr:.3f}', 
               va='center', ha='left' if corr >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'feature_importance_preview.png')
    
    # save feature importance data
    top_features.to_csv('/home/yeblad/Desktop/New_WR_analysis/outputs/feature_correlations.csv', index=False)
    print("feature correlations saved to feature_correlations.csv")

def generate_eda_summary_report(df: pd.DataFrame):
    """generate comprehensive eda summary report"""
    print("generating eda summary report...")
    
    report_lines = []
    report_lines.append("# exploratory data analysis summary report\n\n")
    
    # dataset overview
    report_lines.append("## dataset overview\n")
    report_lines.append(f"- **total records:** {len(df):,}\n")
    report_lines.append(f"- **total features:** {len(df.columns)}\n")
    report_lines.append(f"- **numeric features:** {len(df.select_dtypes(include=[np.number]).columns)}\n")
    report_lines.append(f"- **categorical features:** {len(df.select_dtypes(include=['object']).columns)}\n")
    
    # target variable analysis
    if 'has_1000_yard_season' in df.columns:
        target_dist = df['has_1000_yard_season'].value_counts()
        success_rate = target_dist.get(1, 0) / len(df) * 100
        
        report_lines.append("\n## target variable analysis\n")
        report_lines.append(f"- **overall success rate:** {success_rate:.1f}%\n")
        report_lines.append(f"- **successful players:** {target_dist.get(1, 0):,}\n")
        report_lines.append(f"- **unsuccessful players:** {target_dist.get(0, 0):,}\n")
        
        # class imbalance ratio
        imbalance_ratio = target_dist.get(0, 0) / max(target_dist.get(1, 1), 1)
        report_lines.append(f"- **class imbalance ratio:** {imbalance_ratio:.1f}:1\n")
    
    # draft analysis
    if 'draft_round' in df.columns:
        draft_analysis = df.groupby('draft_round').agg({
            'has_1000_yard_season': ['count', 'sum', 'mean']
        }).round(3)
        draft_analysis.columns = ['total', 'successful', 'success_rate']
        
        report_lines.append("\n## draft position analysis\n")
        report_lines.append("| round | total | successful | success rate |\n")
        report_lines.append("|-------|-------|------------|-------------|\n")
        
        for round_num, row in draft_analysis.iterrows():
            if row['total'] >= 5:  # minimum sample size
                report_lines.append(f"| {round_num} | {row['total']} | {row['successful']} | {row['success_rate']*100:.1f}% |\n")
    
    # performance metrics summary
    performance_cols = ['rec', 'rec_yards', 'rec_td', 'targets', 'catch_rate']
    available_performance = [col for col in performance_cols if col in df.columns]
    
    if available_performance and 'has_1000_yard_season' in df.columns:
        report_lines.append("\n## rookie performance comparison\n")
        report_lines.append("| metric | successful avg | unsuccessful avg | difference |\n")
        report_lines.append("|--------|----------------|------------------|------------|\n")
        
        for col in available_performance:
            success_avg = df[df['has_1000_yard_season'] == 1][col].mean()
            no_success_avg = df[df['has_1000_yard_season'] == 0][col].mean()
            
            if not (np.isnan(success_avg) or np.isnan(no_success_avg)):
                diff = success_avg - no_success_avg
                report_lines.append(f"| {col.replace('_', ' ')} | {success_avg:.2f} | {no_success_avg:.2f} | {diff:.2f} |\n")
    
    # missing data summary
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        report_lines.append("\n## high missing data features (>50%)\n")
        for col, pct in high_missing.head(10).items():
            report_lines.append(f"- **{col}:** {pct:.1f}% missing\n")
    
    # key insights
    report_lines.append("\n## key insights\n")
    
    if 'draft_round' in df.columns and 'has_1000_yard_season' in df.columns:
        early_round_success = df[df['draft_round'] <= 2]['has_1000_yard_season'].mean() * 100
        late_round_success = df[df['draft_round'] >= 5]['has_1000_yard_season'].mean() * 100
        
        report_lines.append(f"- early round picks (1-2) have {early_round_success:.1f}% success rate\n")
        report_lines.append(f"- late round picks (5+) have {late_round_success:.1f}% success rate\n")
    
    if 'rec_yards' in df.columns and 'has_1000_yard_season' in df.columns:
        high_rookie_yards = df[df['rec_yards'] >= 500]['has_1000_yard_season'].mean() * 100
        low_rookie_yards = df[df['rec_yards'] < 200]['has_1000_yard_season'].mean() * 100
        
        report_lines.append(f"- rookies with 500+ yards have {high_rookie_yards:.1f}% success rate\n")
        report_lines.append(f"- rookies with <200 yards have {low_rookie_yards:.1f}% success rate\n")
    
    # save report
    report_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/eda_summary_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"eda summary report saved to: {report_path}")

def main():
    """main function to execute exploratory data analysis"""
    print("starting exploratory data analysis")
    print("="*50)
    
    # load cleaned data
    df = load_cleaned_data()
    
    if df.empty:
        print("no data available for analysis")
        return
    
    # setup plotting style
    setup_plotting_style()
    
    # create all visualizations
    create_target_distribution_plot(df)
    create_draft_analysis_plots(df)
    create_rookie_performance_plots(df)
    create_correlation_heatmap(df)
    create_time_trend_analysis(df)
    create_outlier_analysis(df)
    create_feature_importance_preview(df)
    
    # generate summary report
    generate_eda_summary_report(df)
    
    print("\nexploratory data analysis completed successfully!")
    print("all visualizations saved to /home/yeblad/Desktop/New_WR_analysis/figs/")
    print("analysis reports saved to /home/yeblad/Desktop/New_WR_analysis/outputs/")

if __name__ == "__main__":
    main()
