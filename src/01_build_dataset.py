"""
data integration and cleaning module for nfl wide receiver rookie prediction
combines all raw csv files, handles missing values, and creates unified dataset
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils import (
    standardize_player_name, standardize_team_name, get_column_mapping,
    clean_numeric_column, print_data_summary
)

def load_receiving_data() -> pd.DataFrame:
    """load and combine all receiving statistics files (rec20XX.csv)"""
    print("loading receiving statistics data...")
    
    rec_files = glob.glob('/home/yeblad/Desktop/New_WR_analysis/Uploads/rec20*.csv')
    rec_files.sort()
    
    all_rec_data = []
    
    for file in rec_files:
        year = int(file.split('rec')[1].split('.')[0])
        print(f"  processing {file} (year: {year})")
        
        try:
            df = pd.read_csv(file)
            df['season'] = year
            
            # standardize column names
            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)
            
            # clean player names and team names
            if 'player_name' in df.columns:
                df['player_name'] = df['player_name'].apply(standardize_player_name)
            if 'team' in df.columns:
                df['team'] = df['team'].apply(standardize_team_name)
            
            all_rec_data.append(df)
            
        except Exception as e:
            print(f"  error loading {file}: {e}")
    
    if all_rec_data:
        combined_rec = pd.concat(all_rec_data, ignore_index=True, sort=False)
        print(f"combined receiving data: {combined_rec.shape}")
        return combined_rec
    else:
        return pd.DataFrame()

def load_rookie_data() -> pd.DataFrame:
    """load and combine all rookie draft information files"""
    print("loading rookie draft data...")
    
    rookie_files = glob.glob('/home/yeblad/Desktop/New_WR_analysis/Uploads/*rookie*.csv')
    rookie_files.extend(glob.glob('/home/yeblad/Desktop/New_WR_analysis/Uploads/20*rookies.csv'))
    rookie_files = list(set(rookie_files))  # remove duplicates
    rookie_files.sort()
    
    all_rookie_data = []
    
    for file in rookie_files:
        print(f"  processing {file}")
        
        try:
            # try different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"  could not read {file} with any encoding")
                continue
            
            # extract year from filename
            year_match = None
            for part in file.split('/'):
                if any(char.isdigit() for char in part):
                    year_str = ''.join(filter(str.isdigit, part))
                    if len(year_str) == 4 and year_str.startswith('20'):
                        year_match = int(year_str)
                        break
            
            if year_match:
                df['rookie_year'] = year_match
            
            # standardize column names
            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)
            
            # filter for wide receivers only
            if 'Pos' in df.columns:
                df = df[df['Pos'] == 'WR'].copy()
            elif 'position' in df.columns:
                df = df[df['position'] == 'WR'].copy()
            
            # clean player names and team names
            if 'player_name' in df.columns:
                df['player_name'] = df['player_name'].apply(standardize_player_name)
            if 'team' in df.columns:
                df['team'] = df['team'].apply(standardize_team_name)
            
            all_rookie_data.append(df)
            
        except Exception as e:
            print(f"  error loading {file}: {e}")
    
    if all_rookie_data:
        combined_rookies = pd.concat(all_rookie_data, ignore_index=True, sort=False)
        
        # remove duplicates based on player name and rookie year
        if 'player_name' in combined_rookies.columns and 'rookie_year' in combined_rookies.columns:
            combined_rookies = combined_rookies.drop_duplicates(
                subset=['player_name', 'rookie_year'], keep='first'
            )
        
        print(f"combined rookie data: {combined_rookies.shape}")
        return combined_rookies
    else:
        return pd.DataFrame()

def load_advanced_metrics() -> pd.DataFrame:
    """load and combine advanced wr metrics files"""
    print("loading advanced metrics data...")
    
    advanced_files = glob.glob('/home/yeblad/Desktop/New_WR_analysis/Uploads/nfl-advanced-wr-*.csv')
    advanced_files.sort()
    
    all_advanced_data = []
    
    for file in advanced_files:
        print(f"  processing {file}")
        
        try:
            df = pd.read_csv(file)
            
            # extract year from filename
            year_match = None
            for part in file.split('-'):
                if part.isdigit() and len(part) == 4:
                    year_match = int(part.split('.')[0])
                    break
            
            if year_match:
                df['season'] = year_match
            
            # standardize column names
            column_mapping = get_column_mapping()
            df = df.rename(columns=column_mapping)
            
            # clean player names and team names
            if 'player' in df.columns:
                df['player_name'] = df['player'].apply(standardize_player_name)
            if 'team' in df.columns:
                df['team'] = df['team'].apply(standardize_team_name)
            
            all_advanced_data.append(df)
            
        except Exception as e:
            print(f"  error loading {file}: {e}")
    
    if all_advanced_data:
        combined_advanced = pd.concat(all_advanced_data, ignore_index=True, sort=False)
        print(f"combined advanced metrics: {combined_advanced.shape}")
        return combined_advanced
    else:
        return pd.DataFrame()

def load_target_data() -> pd.DataFrame:
    """load 1000+ yard seasons target data"""
    print("loading target variable data...")
    
    try:
        df = pd.read_csv('/home/yeblad/Desktop/New_WR_analysis/Uploads/1kseasons.csv')
        
        # standardize column names
        df = df.rename(columns={
            'Player': 'player_name',
            'Rookie year': 'rookie_year',
            '1,000 Yard Seasons': 'thousand_yard_seasons'
        })
        
        # clean player names
        df['player_name'] = df['player_name'].apply(standardize_player_name)
        
        # clean numeric columns
        df['rookie_year'] = clean_numeric_column(df['rookie_year'])
        df['thousand_yard_seasons'] = clean_numeric_column(df['thousand_yard_seasons'])
        
        # create binary target variable
        df['has_1000_yard_season'] = (df['thousand_yard_seasons'] > 0).astype(int)
        
        print(f"target data loaded: {df.shape}")
        return df
        
    except Exception as e:
        print(f"error loading target data: {e}")
        return pd.DataFrame()

def merge_all_datasets(rec_data: pd.DataFrame, rookie_data: pd.DataFrame, 
                      advanced_data: pd.DataFrame, target_data: pd.DataFrame) -> pd.DataFrame:
    """merge all datasets into unified dataframe"""
    print("merging all datasets...")
    
    # start with rookie data as base
    merged_df = rookie_data.copy()
    print(f"starting with rookie data: {merged_df.shape}")
    
    # merge with receiving data for rookie seasons
    if not rec_data.empty and 'player_name' in merged_df.columns and 'rookie_year' in merged_df.columns:
        # create merge key for rookie seasons
        rec_data_rookie = rec_data.copy()
        if 'player_name' in rec_data_rookie.columns and 'season' in rec_data_rookie.columns:
            rec_data_rookie['merge_key'] = rec_data_rookie['player_name'] + '_' + rec_data_rookie['season'].astype(str)
            
            merged_df['merge_key'] = merged_df['player_name'] + '_' + merged_df['rookie_year'].astype(str)
            
            merged_df = pd.merge(
                merged_df, rec_data_rookie,
                on='merge_key', 
                how='left',
                suffixes=('', '_rec')
            )
            merged_df = merged_df.drop('merge_key', axis=1)
            print(f"after merging receiving data: {merged_df.shape}")
    
    # merge with advanced metrics
    if not advanced_data.empty and 'player_name' in merged_df.columns and 'rookie_year' in merged_df.columns:
        # create merge key for advanced metrics
        advanced_data_copy = advanced_data.copy()
        if 'player_name' in advanced_data_copy.columns and 'season' in advanced_data_copy.columns:
            advanced_data_copy['merge_key'] = advanced_data_copy['player_name'] + '_' + advanced_data_copy['season'].astype(str)
            
            merged_df['merge_key'] = merged_df['player_name'] + '_' + merged_df['rookie_year'].astype(str)
            
            merged_df = pd.merge(
                merged_df, advanced_data_copy,
                on='merge_key',
                how='left',
                suffixes=('', '_adv')
            )
            merged_df = merged_df.drop('merge_key', axis=1)
            print(f"after merging advanced metrics: {merged_df.shape}")
        else:
            print("advanced metrics missing required columns for merge")
    
    # merge with target data
    if not target_data.empty and 'player_name' in merged_df.columns and 'rookie_year' in merged_df.columns:
        target_cols = ['player_name', 'rookie_year', 'has_1000_yard_season', 'thousand_yard_seasons']
        available_target_cols = [col for col in target_cols if col in target_data.columns]
        
        merged_df = pd.merge(
            merged_df, target_data[available_target_cols],
            on=['player_name', 'rookie_year'],
            how='left'
        )
        print(f"after merging target data: {merged_df.shape}")
    
    return merged_df

def clean_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """comprehensive cleaning of merged dataset"""
    print("cleaning merged dataset...")
    
    df_clean = df.copy()
    
    # remove rows with missing player names
    if 'player_name' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['player_name'])
    
    # clean numeric columns
    numeric_columns = [
        'draft_round', 'draft_pick', 'age', 'rec', 'rec_yards', 'rec_td',
        'targets', 'routes_run', 'air_yards', 'yac', 'target_share',
        'route_participation', 'contested_catch_pct'
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = clean_numeric_column(df_clean[col])
    
    # handle missing target variable
    if 'has_1000_yard_season' not in df_clean.columns:
        df_clean['has_1000_yard_season'] = 0
    else:
        df_clean['has_1000_yard_season'] = df_clean['has_1000_yard_season'].fillna(0)
    
    # create derived features
    if 'rec' in df_clean.columns and 'targets' in df_clean.columns:
        df_clean['catch_rate'] = np.where(
            df_clean['targets'] > 0,
            df_clean['rec'] / df_clean['targets'],
            0
        )
    
    if 'rec_yards' in df_clean.columns and 'rec' in df_clean.columns:
        df_clean['yards_per_reception'] = np.where(
            df_clean['rec'] > 0,
            df_clean['rec_yards'] / df_clean['rec'],
            0
        )
    
    if 'rec_yards' in df_clean.columns and 'routes_run' in df_clean.columns:
        df_clean['yards_per_route_run'] = np.where(
            df_clean['routes_run'] > 0,
            df_clean['rec_yards'] / df_clean['routes_run'],
            0
        )
    
    # remove duplicate rows
    if 'player_name' in df_clean.columns and 'rookie_year' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['player_name', 'rookie_year'], keep='first')
    
    print(f"cleaned dataset shape: {df_clean.shape}")
    return df_clean

def main():
    """main function to execute data integration pipeline"""
    print("starting nfl wr rookie prediction data integration pipeline")
    print("="*60)
    
    # load all datasets
    rec_data = load_receiving_data()
    rookie_data = load_rookie_data()
    advanced_data = load_advanced_metrics()
    target_data = load_target_data()
    
    # debug: print column names
    print("\ndebugging column names:")
    print(f"rookie data columns: {list(rookie_data.columns) if not rookie_data.empty else 'empty'}")
    print(f"advanced data columns: {list(advanced_data.columns) if not advanced_data.empty else 'empty'}")
    
    # merge datasets
    merged_data = merge_all_datasets(rec_data, rookie_data, advanced_data, target_data)
    
    # clean merged dataset
    clean_data = clean_merged_dataset(merged_data)
    
    # print summary statistics
    print_data_summary(clean_data, "final cleaned dataset")
    
    # save cleaned dataset
    output_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/cleaned_dataset.parquet'
    clean_data.to_parquet(output_path, index=False)
    print(f"\ncleaned dataset saved to: {output_path}")
    
    # save csv version for inspection
    csv_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/cleaned_dataset.csv'
    clean_data.to_csv(csv_path, index=False)
    print(f"csv version saved to: {csv_path}")
    
    # create data profile report
    create_data_profile(clean_data)
    
    print("\ndata integration pipeline completed successfully!")

def create_data_profile(df: pd.DataFrame):
    """create basic data profiling report"""
    print("creating data profile report...")
    
    profile_lines = []
    profile_lines.append("# nfl wr rookie prediction - data profile report\n")
    profile_lines.append(f"**dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns\n")
    profile_lines.append(f"**memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} mb\n")
    
    # target variable distribution
    if 'has_1000_yard_season' in df.columns:
        target_dist = df['has_1000_yard_season'].value_counts()
        profile_lines.append("\n## target variable distribution\n")
        profile_lines.append(f"- no 1000+ yard seasons: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df)*100:.1f}%)\n")
        profile_lines.append(f"- has 1000+ yard seasons: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df)*100:.1f}%)\n")
    
    # missing values summary
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    }).sort_values('missing_count', ascending=False)
    
    profile_lines.append("\n## missing values summary\n")
    for col, row in missing_df[missing_df['missing_count'] > 0].iterrows():
        profile_lines.append(f"- {col}: {row['missing_count']} ({row['missing_percent']:.1f}%)\n")
    
    # numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile_lines.append("\n## numeric columns summary\n")
        desc = df[numeric_cols].describe()
        profile_lines.append(desc.to_string())
        profile_lines.append("\n")
    
    # save profile report
    profile_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/data_profile_report.md'
    with open(profile_path, 'w') as f:
        f.writelines(profile_lines)
    
    print(f"data profile report saved to: {profile_path}")

if __name__ == "__main__":
    main()
