
"""
model interpretation module for nfl wide receiver rookie prediction
provides shap analysis, feature importance, and model explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import joblib
import shap
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils import save_figure, setup_plotting_style

def load_model_and_data() -> Tuple[Any, pd.DataFrame, pd.Series]:
    """load trained model and data for interpretation"""
    try:
        # load best model
        model = joblib.load('/home/yeblad/Desktop/New_WR_analysis/outputs/best_model.pkl')
        
        # load features and target
        X = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/features_X.parquet')
        y_df = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/target_y.parquet')
        y = y_df['target']
        
        print(f"loaded model: {type(model)}")
        print(f"loaded features: {X.shape}")
        print(f"loaded target: {y.shape}")
        
        return model, X, y
        
    except Exception as e:
        print(f"error loading model and data: {e}")
        return None, pd.DataFrame(), pd.Series()

def calculate_feature_importance(model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """calculate multiple types of feature importance"""
    print("calculating feature importance...")
    
    importance_results = []
    
    # try to get built-in feature importance
    try:
        if hasattr(model, 'feature_importances_'):
            # for tree-based models
            importances = model.feature_importances_
        elif hasattr(model.named_steps['model'], 'feature_importances_'):
            # for pipeline with tree-based models
            importances = model.named_steps['model'].feature_importances_
        elif hasattr(model.named_steps['model'], 'coef_'):
            # for linear models
            importances = np.abs(model.named_steps['model'].coef_[0])
        else:
            importances = None
        
        if importances is not None:
            for i, importance in enumerate(importances):
                importance_results.append({
                    'feature': X.columns[i],
                    'importance_builtin': importance,
                    'rank_builtin': 0
                })
    except Exception as e:
        print(f"built-in importance failed: {e}")
    
    # permutation importance
    try:
        perm_importance = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, n_jobs=-1
        )
        
        for i, (importance, std) in enumerate(zip(perm_importance.importances_mean, perm_importance.importances_std)):
            if i < len(importance_results):
                importance_results[i]['importance_permutation'] = importance
                importance_results[i]['importance_permutation_std'] = std
            else:
                importance_results.append({
                    'feature': X.columns[i],
                    'importance_permutation': importance,
                    'importance_permutation_std': std
                })
    except Exception as e:
        print(f"permutation importance failed: {e}")
    
    # create dataframe and rank features
    importance_df = pd.DataFrame(importance_results)
    
    if 'importance_builtin' in importance_df.columns:
        importance_df['rank_builtin'] = importance_df['importance_builtin'].rank(ascending=False)
    
    if 'importance_permutation' in importance_df.columns:
        importance_df['rank_permutation'] = importance_df['importance_permutation'].rank(ascending=False)
    
    # sort by best available importance measure
    if 'importance_permutation' in importance_df.columns:
        importance_df = importance_df.sort_values('importance_permutation', ascending=False)
    elif 'importance_builtin' in importance_df.columns:
        importance_df = importance_df.sort_values('importance_builtin', ascending=False)
    
    print(f"calculated importance for {len(importance_df)} features")
    return importance_df

def create_shap_analysis(model: Any, X: pd.DataFrame, sample_size: int = 100) -> Tuple[Any, np.ndarray]:
    """create shap analysis for model interpretation"""
    print("creating shap analysis...")
    
    # sample data for efficiency
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    try:
        # determine explainer type based on model
        model_type = str(type(model)).lower()
        
        if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'gradient' in model_type:
            # tree-based explainer
            if hasattr(model, 'named_steps'):
                # pipeline - need to transform data first
                X_transformed = model.named_steps['preprocessor'].transform(X_sample)
                if hasattr(model.named_steps, 'sampler'):
                    # skip sampler for explanation
                    explainer = shap.TreeExplainer(model.named_steps['model'])
                else:
                    explainer = shap.TreeExplainer(model.named_steps['model'])
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
        else:
            # kernel explainer for other models
            explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:10])
            shap_values = explainer.shap_values(X_sample)
        
        # handle different shap value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class for binary classification
        
        print(f"shap analysis completed for {len(X_sample)} samples")
        return explainer, shap_values
        
    except Exception as e:
        print(f"shap analysis failed: {e}")
        return None, None

def create_feature_importance_plots(importance_df: pd.DataFrame) -> None:
    """create feature importance visualization plots"""
    print("creating feature importance plots...")
    
    setup_plotting_style()
    
    # top features plot
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # built-in importance
    if 'importance_builtin' in top_features.columns:
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(top_features)), top_features['importance_builtin'], alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
        ax1.set_xlabel('Built-in Importance')
        ax1.set_title('Top 20 Features - Built-in Importance')
        ax1.invert_yaxis()
        
        # add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # permutation importance
    if 'importance_permutation' in top_features.columns:
        ax2 = axes[1]
        bars2 = ax2.barh(range(len(top_features)), top_features['importance_permutation'], 
                        xerr=top_features.get('importance_permutation_std', 0), alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'].str.replace('_', ' ').str.title())
        ax2.set_xlabel('Permutation Importance')
        ax2.set_title('Top 20 Features - Permutation Importance')
        ax2.invert_yaxis()
        
        # add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # hide empty subplot if needed
    if 'importance_builtin' not in top_features.columns:
        axes[0].set_visible(False)
    if 'importance_permutation' not in top_features.columns:
        axes[1].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'feature_importance.png')
    
    # importance comparison plot
    if 'importance_builtin' in importance_df.columns and 'importance_permutation' in importance_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(importance_df['importance_builtin'], importance_df['importance_permutation'], 
                  alpha=0.6, s=50)
        
        # add diagonal line
        max_val = max(importance_df['importance_builtin'].max(), importance_df['importance_permutation'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Built-in Importance')
        ax.set_ylabel('Permutation Importance')
        ax.set_title('Feature Importance Comparison')
        ax.grid(True, alpha=0.3)
        
        # annotate top features
        top_5 = importance_df.head(5)
        for _, row in top_5.iterrows():
            ax.annotate(row['feature'].replace('_', ' '), 
                       (row['importance_builtin'], row['importance_permutation']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        save_figure(fig, 'importance_comparison.png')

def create_shap_plots(explainer: Any, shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """create shap visualization plots"""
    print("creating shap plots...")
    
    if explainer is None or shap_values is None:
        print("shap analysis not available, skipping plots")
        return
    
    try:
        # summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        save_figure(fig, 'shap_summary.png')
        
        # detailed summary plot
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.title('SHAP Feature Impact Summary')
        plt.tight_layout()
        save_figure(fig, 'shap_detailed_summary.png')
        
        # waterfall plot for a high-probability prediction
        if len(shap_values) > 0:
            # find a positive prediction example
            model = joblib.load('/home/yeblad/Desktop/New_WR_analysis/outputs/best_model.pkl')
            probabilities = model.predict_proba(X)[:, 1]
            high_prob_idx = np.argmax(probabilities)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[high_prob_idx], 
                               base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                               data=X.iloc[high_prob_idx]),
                show=False,
                max_display=15
            )
            plt.title(f'SHAP Waterfall Plot - High Probability Example (P={probabilities[high_prob_idx]:.3f})')
            plt.tight_layout()
            save_figure(fig, 'shap_waterfall_high.png')
            
            # waterfall plot for a low-probability prediction
            low_prob_idx = np.argmin(probabilities)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[low_prob_idx], 
                               base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                               data=X.iloc[low_prob_idx]),
                show=False,
                max_display=15
            )
            plt.title(f'SHAP Waterfall Plot - Low Probability Example (P={probabilities[low_prob_idx]:.3f})')
            plt.tight_layout()
            save_figure(fig, 'shap_waterfall_low.png')
        
    except Exception as e:
        print(f"shap plotting failed: {e}")

def analyze_feature_interactions(model: Any, X: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    """analyze interactions between top features"""
    print("analyzing feature interactions...")
    
    # get top features for interaction analysis
    top_features = importance_df.head(10)['feature'].tolist()
    
    if len(top_features) < 2:
        print("insufficient features for interaction analysis")
        return
    
    try:
        # create interaction plots for top feature pairs
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        probabilities = model.predict_proba(X)[:, 1]
        
        plot_count = 0
        for i in range(min(4, len(top_features)-1)):
            if plot_count >= 4:
                break
                
            feature1 = top_features[i]
            feature2 = top_features[i+1]
            
            ax = axes[plot_count]
            
            # create scatter plot with probability as color
            scatter = ax.scatter(X[feature1], X[feature2], c=probabilities, 
                               cmap='RdYlBu_r', alpha=0.6, s=30)
            
            ax.set_xlabel(feature1.replace('_', ' ').title())
            ax.set_ylabel(feature2.replace('_', ' ').title())
            ax.set_title(f'Feature Interaction: {feature1} vs {feature2}')
            
            # add colorbar
            plt.colorbar(scatter, ax=ax, label='Success Probability')
            
            plot_count += 1
        
        # hide unused subplots
        for i in range(plot_count, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        save_figure(fig, 'feature_interactions.png')
        
    except Exception as e:
        print(f"feature interaction analysis failed: {e}")

def create_prediction_analysis(model: Any, X: pd.DataFrame, y: pd.Series) -> None:
    """analyze model predictions and create diagnostic plots"""
    print("creating prediction analysis...")
    
    try:
        # get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # probability distribution
        ax1 = axes[0, 0]
        ax1.hist(probabilities[y == 0], bins=30, alpha=0.6, label='No Success', density=True)
        ax1.hist(probabilities[y == 1], bins=30, alpha=0.6, label='Success', density=True)
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Probability Distribution by Actual Outcome')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # prediction confidence
        ax2 = axes[0, 1]
        confidence = np.maximum(probabilities, 1 - probabilities)
        ax2.hist(confidence, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Model Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        # probability vs actual outcome
        ax3 = axes[1, 0]
        prob_bins = np.linspace(0, 1, 11)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        
        actual_rates = []
        for i in range(len(prob_bins) - 1):
            mask = (probabilities >= prob_bins[i]) & (probabilities < prob_bins[i+1])
            if mask.sum() > 0:
                actual_rates.append(y[mask].mean())
            else:
                actual_rates.append(0)
        
        ax3.plot(bin_centers, actual_rates, 'o-', label='Actual Rate', linewidth=2)
        ax3.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', alpha=0.7)
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Actual Success Rate')
        ax3.set_title('Calibration Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # residual analysis
        ax4 = axes[1, 1]
        residuals = y - probabilities
        ax4.scatter(probabilities, residuals, alpha=0.6, s=30)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Residuals (Actual - Predicted)')
        ax4.set_title('Residual Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, 'prediction_analysis.png')
        
    except Exception as e:
        print(f"prediction analysis failed: {e}")

def save_interpretation_results(importance_df: pd.DataFrame, shap_values: np.ndarray = None) -> None:
    """save interpretation results to files"""
    print("saving interpretation results...")
    
    # save feature importance
    importance_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"feature importance saved to: {importance_path}")
    
    # save top features summary
    top_features = importance_df.head(20)
    
    summary_lines = []
    summary_lines.append("# model interpretation summary\n\n")
    summary_lines.append("## top 20 most important features\n\n")
    summary_lines.append("| rank | feature | importance | description |\n")
    summary_lines.append("|------|---------|------------|-------------|\n")
    
    importance_col = 'importance_permutation' if 'importance_permutation' in importance_df.columns else 'importance_builtin'
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        feature_name = row['feature']
        importance_val = row.get(importance_col, 0)
        description = get_feature_description(feature_name)
        
        summary_lines.append(f"| {i} | {feature_name} | {importance_val:.4f} | {description} |\n")
    
    # key insights
    summary_lines.append("\n## key insights\n\n")
    
    if len(importance_df) > 0:
        top_feature = importance_df.iloc[0]
        summary_lines.append(f"- **most important feature**: {top_feature['feature']} with importance {top_feature.get(importance_col, 0):.4f}\n")
        
        # feature categories
        draft_features = [f for f in top_features['feature'] if 'draft' in f.lower()]
        performance_features = [f for f in top_features['feature'] if any(perf in f.lower() for perf in ['rec', 'yards', 'td', 'targets'])]
        efficiency_features = [f for f in top_features['feature'] if any(eff in f.lower() for eff in ['rate', 'per', 'efficiency'])]
        
        summary_lines.append(f"- **draft-related features in top 20**: {len(draft_features)}\n")
        summary_lines.append(f"- **performance features in top 20**: {len(performance_features)}\n")
        summary_lines.append(f"- **efficiency features in top 20**: {len(efficiency_features)}\n")
    
    # save summary
    summary_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/interpretation_summary.md'
    with open(summary_path, 'w') as f:
        f.writelines(summary_lines)
    
    print(f"interpretation summary saved to: {summary_path}")

def get_feature_description(feature_name: str) -> str:
    """get description for a feature"""
    descriptions = {
        'rec_yards': 'total receiving yards in rookie season',
        'rec': 'total receptions in rookie season',
        'targets': 'total targets in rookie season',
        'rec_td': 'receiving touchdowns in rookie season',
        'catch_rate': 'receptions divided by targets',
        'yards_per_reception': 'average yards per reception',
        'draft_pick': 'overall draft position',
        'draft_round': 'draft round (1-7)',
        'age': 'age during rookie season',
        'rookie_production_score': 'composite rookie production metric',
        'efficiency_score': 'composite efficiency rating'
    }
    
    # pattern matching for generated features
    if '_plus' in feature_name:
        return f'binary indicator for achieving threshold'
    elif '_x_' in feature_name:
        return f'interaction feature'
    elif '_log' in feature_name:
        return f'log transformation'
    elif '_sqrt' in feature_name:
        return f'square root transformation'
    elif 'score' in feature_name:
        return f'composite score metric'
    
    return descriptions.get(feature_name, 'engineered feature')

def main():
    """main function to execute model interpretation pipeline"""
    print("starting model interpretation analysis")
    print("="*50)
    
    # load model and data
    model, X, y = load_model_and_data()
    
    if model is None or X.empty:
        print("no model or data available for interpretation")
        return
    
    # setup plotting style
    setup_plotting_style()
    
    # calculate feature importance
    importance_df = calculate_feature_importance(model, X, y)
    
    # create feature importance plots
    create_feature_importance_plots(importance_df)
    
    # shap analysis
    explainer, shap_values = create_shap_analysis(model, X, sample_size=100)
    
    # create shap plots
    create_shap_plots(explainer, shap_values, X)
    
    # analyze feature interactions
    analyze_feature_interactions(model, X, importance_df)
    
    # create prediction analysis
    create_prediction_analysis(model, X, y)
    
    # save results
    save_interpretation_results(importance_df, shap_values)
    
    print("\nmodel interpretation completed successfully!")
    print("results saved to /home/yeblad/Desktop/New_WR_analysis/outputs/")
    print("visualizations saved to /home/yeblad/Desktop/New_WR_analysis/figs/")

if __name__ == "__main__":
    main()
