#!/usr/bin/env python3
"""
Comprehensive feature analysis and selection to reduce overfitting
Identifies and removes highly correlated features and features causing overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import joblib

# Set paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figs"

def load_data():
    """Load features and target"""
    print("Loading data...")
    X = pd.read_parquet(OUTPUT_DIR / 'features_X.parquet')
    y = pd.read_parquet(OUTPUT_DIR / 'target_y.parquet')['target']
    df = pd.read_parquet(OUTPUT_DIR / 'cleaned_dataset.parquet')
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df

def analyze_feature_correlations(X, y, threshold=0.9):
    """
    Analyze correlations between features and with target
    Identify features to remove
    """
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # 1. Feature-Target Correlations
    print("\n=== FEATURE-TARGET CORRELATIONS ===")
    target_corrs = {}
    for col in X.columns:
        corr = X[col].corr(y)
        target_corrs[col] = abs(corr)
    
    sorted_target_corrs = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 features by correlation with target:")
    high_target_corr_features = []
    for feat, corr in sorted_target_corrs[:15]:
        if corr > 0.7:
            print(f"WARNING: {feat:<30} {corr:.3f} - VERY HIGH (potential overfit)")
            high_target_corr_features.append(feat)
        elif corr > 0.5:
            print(f"!! {feat:<30} {corr:.3f} - High")
        else:
            print(f"    {feat:<30} {corr:.3f}")
    
    # 2. Feature-Feature Correlations
    print("\n=== FEATURE-FEATURE CORRELATIONS ===")
    corr_matrix = X.corr().abs()
    
    # Find highly correlated feature pairs
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = []
    for column in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, column] >= threshold:
                high_corr_pairs.append((row, column, upper_tri.loc[row, column]))
    
    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} feature pairs with correlation >= {threshold}:")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {feat1:<25} <-> {feat2:<25} : {corr:.3f}")
    
    # 3. Identify features to remove
    features_to_remove = set()
    
    # Remove one from each highly correlated pair (keep the one with higher target correlation)
    for feat1, feat2, _ in high_corr_pairs:
        if target_corrs[feat1] < target_corrs[feat2]:
            features_to_remove.add(feat1)
        else:
            features_to_remove.add(feat2)
    
    # Consider removing features with extremely high target correlation
    for feat in high_target_corr_features:
        if target_corrs[feat] > 0.75:  # Very high threshold
            print(f"\nConsidering removing '{feat}' due to correlation {target_corrs[feat]:.3f} with target")
    
    # Create correlation heatmap
    plt.figure(figsize=(20, 16))
    
    # Select top features for visualization
    top_features = [feat for feat, _ in sorted_target_corrs[:30]]
    corr_subset = X[top_features].corr()
    
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=False, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Top 30 Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\nCorrelation matrix saved to {FIG_DIR / 'feature_correlation_matrix.png'}")
    plt.close()
    
    return features_to_remove, target_corrs, high_corr_pairs

def analyze_feature_variance(X):
    """Analyze feature variance and identify low-variance features"""
    print("\n=== FEATURE VARIANCE ANALYSIS ===")
    
    # Calculate variance
    variances = X.var()
    sorted_vars = variances.sort_values()
    
    print("\nLowest variance features (potential candidates for removal):")
    for feat in sorted_vars.head(10).index:
        print(f"  {feat:<30} variance: {sorted_vars[feat]:.6f}")
    
    # Features with near-zero variance
    near_zero_var = sorted_vars[sorted_vars < 0.01]
    if len(near_zero_var) > 0:
        print(f"\nWARNING: Found {len(near_zero_var)} features with near-zero variance")
        return list(near_zero_var.index)
    
    return []

def mutual_information_analysis(X, y):
    """Calculate mutual information scores"""
    print("\n=== MUTUAL INFORMATION ANALYSIS ===")
    
    # Scale features for MI calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("\nTop 15 features by Mutual Information:")
    for idx, row in mi_df.head(15).iterrows():
        print(f"  {row['feature']:<30} MI: {row['mi_score']:.4f}")
    
    # Features with very low MI
    low_mi = mi_df[mi_df['mi_score'] < 0.01]
    if len(low_mi) > 0:
        print(f"\nWARNING: Found {len(low_mi)} features with very low MI scores")
    
    return mi_df

def test_feature_subsets(X, y, df):
    """Test different feature subsets to find optimal set"""
    print("\n" + "="*60)
    print("TESTING FEATURE SUBSETS")
    print("="*60)
    
    results = {}
    
    # Define different feature subsets
    subsets = {
        'all_features': list(X.columns),
        'no_rec': [col for col in X.columns if 'rec' not in col.lower()],
        'basic_only': ['rec_yards', 'rec_td', 'targets', 'draft_round', 'draft_pick', 
                       'age', 'catch_rate', 'yards_per_reception'],
        'no_high_corr': None,  # Will be defined after correlation analysis
        'top_mi': None,  # Will be defined after MI analysis
    }
    
    # Get features to remove based on correlation
    features_to_remove, target_corrs, _ = analyze_feature_correlations(X, y, threshold=0.85)
    subsets['no_high_corr'] = [col for col in X.columns if col not in features_to_remove]
    
    # Get top MI features
    mi_df = mutual_information_analysis(X, y)
    subsets['top_mi'] = list(mi_df.head(20)['feature'])
    
    # Remove the most correlated feature with target if > 0.75
    most_correlated = max(target_corrs.items(), key=lambda x: x[1])
    if most_correlated[1] > 0.75:
        subsets['no_dominant'] = [col for col in X.columns if col != most_correlated[0]]
    
    # Test each subset
    for subset_name, feature_list in subsets.items():
        if feature_list is None or len(feature_list) == 0:
            continue
            
        # Filter to available features
        available_features = [f for f in feature_list if f in X.columns]
        if len(available_features) == 0:
            continue
            
        X_subset = X[available_features]
        
        print(f"\n=== Testing: {subset_name} ({len(available_features)} features) ===")
        
        # Simple model for testing
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,  # Shallow trees to reduce overfitting
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=3
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_subset, y, cv=5, scoring='roc_auc')
        
        # Temporal validation if possible
        temporal_score = None
        if 'rookie_year' in df.columns:
            # Train on pre-2020, test on 2020+
            mask_train = df['rookie_year'] < 2020
            mask_test = df['rookie_year'] >= 2020
            
            if mask_train.sum() > 50 and mask_test.sum() > 20:
                X_train = X_subset[mask_train]
                X_test = X_subset[mask_test]
                y_train = y[mask_train]
                y_test = y[mask_test]
                
                if y_test.sum() > 0:  # Need positive cases
                    model.fit(X_train, y_train)
                    temporal_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        results[subset_name] = {
            'n_features': len(available_features),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'temporal_score': temporal_score
        }
        
        print(f"  CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        if temporal_score:
            print(f"  Temporal ROC AUC: {temporal_score:.3f}")
            print(f"  Generalization gap: {cv_scores.mean() - temporal_score:.3f}")
    
    return results, subsets

def recursive_feature_elimination(X, y):
    """Use RFE to find optimal feature subset"""
    print("\n=== RECURSIVE FEATURE ELIMINATION ===")
    
    # Use a simple model for RFE
    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    
    # RFE with cross-validation
    selector = RFECV(
        estimator,
        step=1,
        cv=5,
        scoring='roc_auc',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    print("Running RFE (this may take a minute)...")
    selector.fit(X, y)
    
    print(f"Optimal number of features: {selector.n_features_}")
    print(f"Best CV score: {selector.cv_results_['mean_test_score'].max():.3f}")
    
    # Get selected features
    selected_features = X.columns[selector.support_].tolist()
    print(f"\nSelected features ({len(selected_features)}):")
    for feat in selected_features:
        print(f"  - {feat}")
    
    return selected_features, selector

def create_optimized_features(X, y, df):
    """Create an optimized feature set based on analysis"""
    print("\n" + "="*60)
    print("CREATING OPTIMIZED FEATURE SET")
    print("="*60)
    
    # Get correlation analysis
    features_to_remove, target_corrs, high_corr_pairs = analyze_feature_correlations(X, y, threshold=0.85)
    
    # Get variance analysis
    low_var_features = analyze_feature_variance(X)
    
    # Get MI scores
    mi_df = mutual_information_analysis(X, y)
    
    # Start with all features
    optimized_features = list(X.columns)
    
    # Remove features step by step
    removed_features = []
    
    # 1. Remove near-zero variance features
    for feat in low_var_features:
        if feat in optimized_features:
            optimized_features.remove(feat)
            removed_features.append((feat, 'near-zero variance'))
    
    # 2. Remove one from highly correlated pairs
    for feat in features_to_remove:
        if feat in optimized_features:
            optimized_features.remove(feat)
            removed_features.append((feat, 'high feature correlation'))
    
    # 3. Remove features with extremely high target correlation
    for feat, corr in target_corrs.items():
        if corr > 0.78 and feat in optimized_features:  # 'rec' is likely here
            optimized_features.remove(feat)
            removed_features.append((feat, f'target correlation {corr:.3f}'))
    
    # 4. Remove features with very low MI
    low_mi_features = mi_df[mi_df['mi_score'] < 0.005]['feature'].tolist()
    for feat in low_mi_features:
        if feat in optimized_features:
            optimized_features.remove(feat)
            removed_features.append((feat, 'low mutual information'))
    
    print(f"\nRemoved {len(removed_features)} features:")
    for feat, reason in removed_features:
        print(f"  - {feat:<30} ({reason})")
    
    print(f"\nOptimized feature set: {len(optimized_features)} features")
    
    # Create the optimized dataset
    X_optimized = X[optimized_features]
    
    # Test the optimized features
    print("\n=== TESTING OPTIMIZED FEATURES ===")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=3
    )
    
    cv_scores = cross_val_score(model, X_optimized, y, cv=5, scoring='roc_auc')
    print(f"Optimized CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Compare with original
    cv_scores_original = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"Original CV ROC AUC: {cv_scores_original.mean():.3f} (+/- {cv_scores_original.std():.3f})")
    
    print(f"Difference: {cv_scores.mean() - cv_scores_original.mean():+.3f}")
    
    return X_optimized, optimized_features, removed_features

def create_visualization_report(results, X, y):
    """Create comprehensive visualization report"""
    print("\n=== CREATING VISUALIZATION REPORT ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature subset comparison
    ax = axes[0, 0]
    subset_names = list(results.keys())
    cv_means = [results[name]['cv_mean'] for name in subset_names]
    cv_stds = [results[name]['cv_std'] for name in subset_names]
    
    x_pos = np.arange(len(subset_names))
    ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset_names, rotation=45, ha='right')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Performance by Feature Subset')
    ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Target')
    ax.legend()
    
    # 2. Temporal vs CV performance
    ax = axes[0, 1]
    temporal_results = [(name, res['cv_mean'], res['temporal_score']) 
                       for name, res in results.items() if res['temporal_score'] is not None]
    
    if temporal_results:
        names, cv_scores, temp_scores = zip(*temporal_results)
        x_pos = np.arange(len(names))
        width = 0.35
        
        ax.bar(x_pos - width/2, cv_scores, width, label='CV Score', alpha=0.7)
        ax.bar(x_pos + width/2, temp_scores, width, label='Temporal Score', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('ROC AUC')
        ax.set_title('CV vs Temporal Validation')
        ax.legend()
    
    # 3. Feature importance distribution
    ax = axes[1, 0]
    # Get feature importances from a simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    ax.bar(range(min(20, len(importances))), importances.head(20).values, alpha=0.7)
    ax.set_xticks(range(min(20, len(importances))))
    ax.set_xticklabels(importances.head(20).index, rotation=45, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title('Top 20 Feature Importances')
    
    # 4. Overfitting analysis
    ax = axes[1, 1]
    gaps = [(name, res['cv_mean'] - res['temporal_score']) 
           for name, res in results.items() if res['temporal_score'] is not None]
    
    if gaps:
        names, gap_values = zip(*gaps)
        colors = ['red' if g > 0.1 else 'yellow' if g > 0.05 else 'green' for g in gap_values]
        ax.bar(range(len(names)), gap_values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('CV - Temporal Gap')
        ax.set_title('Overfitting Analysis (Lower is Better)')
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='High Overfit')
        ax.axhline(y=0.05, color='y', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'feature_selection_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Analysis report saved to {FIG_DIR / 'feature_selection_analysis.png'}")
    plt.close()

def save_results(X_optimized, optimized_features, removed_features, results):
    """Save all results and create summary report"""
    print("\n=== SAVING RESULTS ===")
    
    # Save optimized features
    X_optimized.to_parquet(OUTPUT_DIR / 'features_X_optimized.parquet')
    print(f"Optimized features saved to {OUTPUT_DIR / 'features_X_optimized.parquet'}")
    
    # Save feature lists
    feature_report = {
        'optimized_features': optimized_features,
        'removed_features': removed_features,
        'subset_results': results
    }
    
    import json
    with open(OUTPUT_DIR / 'feature_optimization_report.json', 'w') as f:
        json.dump(feature_report, f, indent=2, default=str)
    
    # Create markdown report
    report = []
    report.append("# Feature Analysis and Selection Report\n")
    report.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"**Original Features**: {len(X_optimized.columns) + len(removed_features)}\n")
    report.append(f"**Optimized Features**: {len(optimized_features)}\n")
    report.append(f"**Features Removed**: {len(removed_features)}\n\n")
    
    report.append("## Removed Features\n")
    for feat, reason in removed_features:
        report.append(f"- **{feat}**: {reason}\n")
    
    report.append("\n## Performance Comparison\n")
    report.append("| Feature Set | CV ROC AUC | Temporal ROC AUC | Gap |\n")
    report.append("|------------|------------|------------------|-----|\n")
    for name, res in results.items():
        if res['temporal_score'] is not None:
            temp = f"{res['temporal_score']:.3f}"
            gap = f"{res['cv_mean'] - res['temporal_score']:.3f}"
        else:
            temp = 'N/A'
            gap = 'N/A'
        report.append(f"| {name} | {res['cv_mean']:.3f} | {temp} | {gap} |\n")
    
    with open(OUTPUT_DIR / 'feature_selection_report.md', 'w') as f:
        f.writelines(report)
    
    print(f"Feature selection report saved to {OUTPUT_DIR / 'feature_selection_report.md'}")

def main():
    """Main execution"""
    print("="*60)
    print("COMPREHENSIVE FEATURE ANALYSIS AND SELECTION")
    print("="*60)
    
    # Load data
    X, y, df = load_data()
    
    # Test different feature subsets
    results, subsets = test_feature_subsets(X, y, df)
    
    # Run RFE
    selected_features_rfe, selector = recursive_feature_elimination(X, y)
    
    # Create optimized feature set
    X_optimized, optimized_features, removed_features = create_optimized_features(X, y, df)
    
    # Add RFE results to comparison
    if selected_features_rfe:
        X_rfe = X[selected_features_rfe]
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                             random_state=42, scale_pos_weight=3)
        cv_scores = cross_val_score(model, X_rfe, y, cv=5, scoring='roc_auc')
        results['rfe_selected'] = {
            'n_features': len(selected_features_rfe),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'temporal_score': None
        }
    
    # Create visualizations
    create_visualization_report(results, X, y)
    
    # Save everything
    save_results(X_optimized, optimized_features, removed_features, results)
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOptimized from {X.shape[1]} to {len(optimized_features)} features")
    print("Check outputs/ folder for detailed reports and optimized features")

if __name__ == "__main__":
    main()
