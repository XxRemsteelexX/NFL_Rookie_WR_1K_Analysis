#!/usr/bin/env python3
"""
Detect and fix data leakage issues in the model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

# Set paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

def check_for_leakage():
    """Check for data leakage in features"""
    
    print("="*60)
    print("DATA LEAKAGE DETECTION")
    print("="*60)
    
    # Load the original dataset
    df = pd.read_parquet(OUTPUT_DIR / 'cleaned_dataset.parquet')
    X = pd.read_parquet(OUTPUT_DIR / 'features_X.parquet')
    y = pd.read_parquet(OUTPUT_DIR / 'target_y.parquet')['target']
    
    print(f"\n=== DATASET INFO ===")
    print(f"Original data shape: {df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Available columns in original data:")
    
    # Check for suspicious columns
    suspicious_cols = []
    
    for col in df.columns:
        if 'thousand_yard' in col.lower():
            print(f"WARNING: SUSPICIOUS: {col} - might contain target information")
            suspicious_cols.append(col)
        if '1000' in col or '1k' in col.lower():
            print(f"WARNING: SUSPICIOUS: {col} - might contain target information")
            suspicious_cols.append(col)
        if 'future' in col.lower() or 'career' in col.lower():
            print(f"WARNING: SUSPICIOUS: {col} - might contain future information")
            suspicious_cols.append(col)
    
    print(f"\n=== TEMPORAL ANALYSIS ===")
    if 'rookie_year' in df.columns:
        years = df['rookie_year'].dropna().sort_values().unique()
        print(f"Years in dataset: {years[:10]} ... {years[-10:]}")
        print(f"Year range: {years.min()} to {years.max()}")
        
        # Check if target is based on future seasons
        print(f"\n=== TARGET VARIABLE ANALYSIS ===")
        print("Checking what 'has_1000_yard_season' actually means...")
        
        # Compare with actual rookie year stats
        if 'rec_yards' in df.columns:
            rookie_1000 = (df['rec_yards'] >= 1000).sum()
            target_1000 = df['has_1000_yard_season'].sum()
            print(f"Rookies with 1000+ yards in rookie year: {rookie_1000}")
            print(f"Positive targets (has_1000_yard_season): {target_1000}")
            
            if target_1000 > rookie_1000:
                print("OK: Target includes FUTURE seasons (correct)")
            else:
                print("WARNING: Target might only be rookie year (incorrect)")
    
    return df, suspicious_cols

def test_temporal_validation(X, y, df):
    """Test model with proper temporal validation"""
    
    print(f"\n=== TEMPORAL VALIDATION ===")
    
    if 'rookie_year' not in df.columns:
        print("No rookie_year column found for temporal validation")
        return
    
    # Align data
    df_with_features = df.copy()
    df_with_features['target'] = y
    
    # Sort by year
    years = df['rookie_year'].dropna()
    
    # Test on different year splits
    test_years = [2018, 2019, 2020, 2021]
    
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    for test_year in test_years:
        mask_train = df['rookie_year'] < test_year
        mask_test = df['rookie_year'] >= test_year
        
        if mask_train.sum() < 50 or mask_test.sum() < 10:
            continue
            
        X_train = X[mask_train]
        X_test = X[mask_test]
        y_train = y[mask_train]
        y_test = y[mask_test]
        
        print(f"\nTrain: years < {test_year} (n={len(X_train)})")
        print(f"Test: years >= {test_year} (n={len(X_test)})")
        
        # Train simple model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        
        if y_test.sum() > 0:  # Need at least one positive case
            test_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"Train ROC AUC: {train_score:.3f}")
            print(f"Test ROC AUC: {test_score:.3f}")
            print(f"Gap: {train_score - test_score:.3f}")
            
            if train_score - test_score > 0.15:
                print("WARNING: Large gap suggests overfitting!")
        else:
            print(f"Train ROC AUC: {train_score:.3f}")
            print("Not enough positive cases in test set")

def check_feature_correlations(X, y):
    """Check if features are too correlated with target"""
    
    print(f"\n=== FEATURE-TARGET CORRELATION ===")
    
    correlations = {}
    for col in X.columns:
        corr = X[col].corr(y)
        correlations[col] = abs(corr)
    
    # Sort by correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 features by correlation with target:")
    for feat, corr in sorted_corrs[:10]:
        if corr > 0.8:
            print(f"WARNING: {feat}: {corr:.3f} - VERY HIGH correlation")
        elif corr > 0.6:
            print(f"!! {feat}: {corr:.3f} - High correlation")
        else:
            print(f"   {feat}: {corr:.3f}")
    
    return correlations

def create_clean_features(df):
    """Create features that definitely don't have leakage"""
    
    print(f"\n=== CREATING CLEAN FEATURES ===")
    
    clean_features = pd.DataFrame()
    
    # Only use rookie year stats
    rookie_stats = ['rec', 'rec_yards', 'rec_td', 'targets', 'age']
    for col in rookie_stats:
        if col in df.columns:
            clean_features[col] = df[col].fillna(0)
    
    # Draft information (known before season)
    draft_cols = ['draft_round', 'draft_pick']
    for col in draft_cols:
        if col in df.columns:
            clean_features[col] = df[col].fillna(999)  # Undrafted = 999
    
    # Create simple derived features
    if 'rec' in clean_features.columns and 'targets' in clean_features.columns:
        clean_features['catch_rate'] = clean_features['rec'] / (clean_features['targets'] + 1)
    
    if 'rec_yards' in clean_features.columns and 'rec' in clean_features.columns:
        clean_features['yards_per_rec'] = clean_features['rec_yards'] / (clean_features['rec'] + 1)
    
    print(f"Created {len(clean_features.columns)} clean features:")
    print(list(clean_features.columns))
    
    return clean_features

def main():
    """Main execution"""
    
    # Check for leakage
    df, suspicious_cols = check_for_leakage()
    
    # Load current features
    X = pd.read_parquet(OUTPUT_DIR / 'features_X.parquet')
    y = pd.read_parquet(OUTPUT_DIR / 'target_y.parquet')['target']
    
    # Check correlations
    correlations = check_feature_correlations(X, y)
    
    # Test temporal validation
    test_temporal_validation(X, y, df)
    
    # Create and test clean features
    clean_X = create_clean_features(df)
    
    if len(clean_X) > 0:
        print(f"\n=== TESTING CLEAN FEATURES ===")
        
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Align indices
        clean_X = clean_X.loc[y.index]
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
        
        scores = cross_val_score(model, clean_X, y, cv=5, scoring='roc_auc')
        print(f"Clean features CV ROC AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Compare with original
        original_model = joblib.load(OUTPUT_DIR / 'best_model.pkl')
        original_scores = cross_val_score(original_model, X, y, cv=5, scoring='roc_auc')
        print(f"Original features CV ROC AUC: {original_scores.mean():.3f}")
        
        if scores.mean() < original_scores.mean() - 0.1:
            print("\nWARNING: Original model is much better - possible leakage!")
        else:
            print("\nOK: Performance similar - leakage unlikely")
        
        # Save clean features
        clean_X.to_parquet(OUTPUT_DIR / 'features_clean.parquet')
        print(f"\nClean features saved to {OUTPUT_DIR / 'features_clean.parquet'}")

if __name__ == "__main__":
    main()
