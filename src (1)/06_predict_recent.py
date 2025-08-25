"""
prediction module for recent nfl wide receiver rookie classes
makes predictions for 2022-2024 rookies with confidence intervals
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
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import save_figure, setup_plotting_style, standardize_player_name

def load_model_and_training_data() -> Tuple[Any, pd.DataFrame, pd.Series]:
    """load trained model and training data"""
    try:
        # load best model
        model = joblib.load('/home/yeblad/Desktop/New_WR_analysis/outputs/best_model.pkl')
        
        # load training features and target
        X_train = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/features_X.parquet')
        y_train_df = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/target_y.parquet')
        y_train = y_train_df['target']
        
        print(f"loaded model: {type(model)}")
        print(f"loaded training data: {X_train.shape}")
        
        return model, X_train, y_train
        
    except Exception as e:
        print(f"error loading model and training data: {e}")
        return None, pd.DataFrame(), pd.Series()

def identify_recent_rookies(df: pd.DataFrame) -> pd.DataFrame:
    """identify and extract recent rookie classes (2022-2024)"""
    print("identifying recent rookie classes...")
    
    # filter for recent years
    recent_years = [2022, 2023, 2024]
    recent_rookies = df[df['rookie_year'].isin(recent_years)].copy()
    
    if len(recent_rookies) == 0:
        print("no recent rookies found, checking all available years")
        available_years = sorted(df['rookie_year'].dropna().unique())
        print(f"available years: {available_years}")
        
        # take the most recent 3 years available
        if len(available_years) >= 3:
            recent_years = available_years[-3:]
            recent_rookies = df[df['rookie_year'].isin(recent_years)].copy()
    
    print(f"found {len(recent_rookies)} recent rookies from years: {sorted(recent_rookies['rookie_year'].unique())}")
    
    # group by year for analysis
    year_counts = recent_rookies['rookie_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} rookies")
    
    return recent_rookies

def create_basic_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """create basic efficiency features for recent rookies"""
    df_eng = df.copy()
    
    # basic efficiency metrics
    if 'rec_yards' in df_eng.columns and 'targets' in df_eng.columns:
        df_eng['yards_per_target'] = np.where(
            df_eng['targets'] > 0,
            df_eng['rec_yards'] / df_eng['targets'],
            0
        )
    
    if 'rec' in df_eng.columns and 'targets' in df_eng.columns:
        df_eng['catch_rate'] = np.where(
            df_eng['targets'] > 0,
            df_eng['rec'] / df_eng['targets'],
            0
        )
    
    if 'rec_yards' in df_eng.columns and 'rec' in df_eng.columns:
        df_eng['yards_per_reception'] = np.where(
            df_eng['rec'] > 0,
            df_eng['rec_yards'] / df_eng['rec'],
            0
        )
    
    # draft capital features
    if 'draft_pick' in df_eng.columns:
        max_pick = 300  # approximate max pick
        df_eng['draft_capital_score'] = np.where(
            df_eng['draft_pick'] > 0,
            (max_pick - df_eng['draft_pick'] + 1) / max_pick,
            0
        )
        df_eng['early_round'] = (df_eng['draft_round'] <= 2).astype(int)
        df_eng['first_round'] = (df_eng['draft_round'] == 1).astype(int)
    
    # age features
    if 'age' in df_eng.columns:
        df_eng['young_rookie'] = (df_eng['age'] <= 21).astype(int)
        df_eng['old_rookie'] = (df_eng['age'] >= 24).astype(int)
    
    # era features
    if 'rookie_year' in df_eng.columns:
        df_eng['modern_era'] = (df_eng['rookie_year'] >= 2011).astype(int)
        df_eng['recent_era'] = (df_eng['rookie_year'] >= 2018).astype(int)
    
    # production thresholds
    if 'rec_yards' in df_eng.columns:
        df_eng['yards_500_plus'] = (df_eng['rec_yards'] >= 500).astype(int)
        df_eng['yards_300_plus'] = (df_eng['rec_yards'] >= 300).astype(int)
    
    if 'rec' in df_eng.columns:
        df_eng['rec_50_plus'] = (df_eng['rec'] >= 50).astype(int)
        df_eng['rec_30_plus'] = (df_eng['rec'] >= 30).astype(int)
    
    if 'targets' in df_eng.columns:
        df_eng['targets_80_plus'] = (df_eng['targets'] >= 80).astype(int)
        df_eng['targets_50_plus'] = (df_eng['targets'] >= 50).astype(int)
    
    # composite scores
    production_features = ['rec', 'rec_yards', 'rec_td', 'targets']
    available_production = [col for col in production_features if col in df_eng.columns]
    
    if len(available_production) >= 3:
        # simple production score
        production_sum = 0
        for col in available_production:
            if col == 'rec_yards':
                production_sum += df_eng[col] / 100  # scale yards
            elif col == 'rec_td':
                production_sum += df_eng[col] * 10  # weight tds higher
            else:
                production_sum += df_eng[col]
        
        df_eng['rookie_production_score'] = production_sum / len(available_production)
    
    # log transformations
    for col in ['rec_yards', 'rec', 'targets']:
        if col in df_eng.columns:
            df_eng[f'{col}_log'] = np.log1p(df_eng[col])
            df_eng[f'{col}_sqrt'] = np.sqrt(df_eng[col])
    
    return df_eng

def prepare_recent_rookie_features(recent_rookies: pd.DataFrame, training_features: pd.DataFrame) -> pd.DataFrame:
    """prepare features for recent rookies using simplified feature engineering"""
    print("preparing features for recent rookies...")
    
    try:
        # apply basic feature engineering
        df_eng = create_basic_efficiency_features(recent_rookies)
        
        # get training feature names
        training_feature_names = training_features.columns.tolist()
        
        # ensure all required features exist
        for feature in training_feature_names:
            if feature not in df_eng.columns:
                # create missing features with reasonable defaults
                if 'score' in feature:
                    df_eng[feature] = 0.5  # neutral score
                elif 'rate' in feature or 'per' in feature:
                    df_eng[feature] = 0.0  # zero rate
                elif '_plus' in feature:
                    df_eng[feature] = 0  # binary threshold not met
                elif 'zscore' in feature:
                    df_eng[feature] = 0.0  # neutral z-score
                else:
                    df_eng[feature] = 0  # default to zero
        
        # select only training features
        X_recent = df_eng[training_feature_names].copy()
        
        # handle missing values same way as training
        for col in X_recent.columns:
            if X_recent[col].dtype in ['float64', 'int64']:
                X_recent[col] = X_recent[col].fillna(0)
            else:
                X_recent[col] = X_recent[col].fillna('unknown')
        
        print(f"prepared features for recent rookies: {X_recent.shape}")
        return X_recent
        
    except Exception as e:
        print(f"feature engineering failed: {e}")
        
        # fallback: use available features and fill missing with 0
        available_features = [col for col in training_features.columns if col in recent_rookies.columns]
        X_recent = pd.DataFrame(index=recent_rookies.index)
        
        # add available features
        for feature in available_features:
            X_recent[feature] = recent_rookies[feature]
        
        # add missing features with default values
        for feature in training_features.columns:
            if feature not in X_recent.columns:
                X_recent[feature] = 0
        
        # reorder to match training
        X_recent = X_recent[training_features.columns]
        
        # handle missing values
        X_recent = X_recent.fillna(0)
        
        print(f"prepared features using fallback method: {X_recent.shape}")
        return X_recent

def make_predictions_with_confidence(model: Any, X_recent: pd.DataFrame, 
                                   recent_rookies: pd.DataFrame) -> pd.DataFrame:
    """make predictions with confidence intervals"""
    print("making predictions with confidence intervals...")
    
    try:
        # get predictions and probabilities
        predictions = model.predict(X_recent)
        probabilities = model.predict_proba(X_recent)[:, 1]
        
        # calculate confidence intervals using bootstrap
        n_bootstrap = 100  # reduced for efficiency
        bootstrap_probs = []
        
        print("calculating bootstrap confidence intervals...")
        for i in range(n_bootstrap):
            # bootstrap sample
            bootstrap_idx = np.random.choice(len(X_recent), len(X_recent), replace=True)
            X_bootstrap = X_recent.iloc[bootstrap_idx]
            
            try:
                bootstrap_prob = model.predict_proba(X_bootstrap)[:, 1]
                bootstrap_probs.append(bootstrap_prob)
            except:
                continue
        
        # calculate confidence intervals
        if bootstrap_probs:
            bootstrap_probs = np.array(bootstrap_probs)
            ci_lower = np.percentile(bootstrap_probs, 2.5, axis=0)
            ci_upper = np.percentile(bootstrap_probs, 97.5, axis=0)
        else:
            # fallback: use normal approximation
            std_error = np.sqrt(probabilities * (1 - probabilities) / len(X_recent))
            ci_lower = probabilities - 1.96 * std_error
            ci_upper = probabilities + 1.96 * std_error
        
        # ensure confidence intervals are valid
        ci_lower = np.maximum(ci_lower, 0)
        ci_upper = np.minimum(ci_upper, 1)
        
        # create results dataframe
        results = pd.DataFrame({
            'player_name': recent_rookies['player_name'].values,
            'rookie_year': recent_rookies['rookie_year'].values,
            'team': recent_rookies.get('team', 'unknown').values,
            'draft_round': recent_rookies.get('draft_round', 0).values,
            'draft_pick': recent_rookies.get('draft_pick', 0).values,
            'rec_yards': recent_rookies.get('rec_yards', 0).values,
            'rec': recent_rookies.get('rec', 0).values,
            'targets': recent_rookies.get('targets', 0).values,
            'prediction': predictions,
            'probability': probabilities,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_width': ci_upper - ci_lower
        })
        
        # add risk categories
        results['risk_category'] = pd.cut(
            results['probability'],
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # sort by probability descending
        results = results.sort_values('probability', ascending=False)
        
        print(f"predictions completed for {len(results)} recent rookies")
        return results
        
    except Exception as e:
        print(f"prediction failed: {e}")
        return pd.DataFrame()

def compare_with_original_model() -> Dict[str, Any]:
    """compare results with original model if available"""
    print("comparing with original model...")
    
    comparison_results = {}
    
    try:
        # try to load original model
        original_model = joblib.load('/home/yeblad/Desktop/New_WR_analysis/Uploads/model.pkl')
        
        # load test data
        X_test = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/features_X.parquet')
        y_test_df = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/target_y.parquet')
        y_test = y_test_df['target']
        
        # get predictions from both models
        new_model = joblib.load('/home/yeblad/Desktop/New_WR_analysis/outputs/best_model.pkl')
        
        # make predictions
        new_pred_prob = new_model.predict_proba(X_test)[:, 1]
        
        # try to get original model predictions
        try:
            original_pred_prob = original_model.predict_proba(X_test)[:, 1]
            
            # calculate metrics for both models
            new_auc = roc_auc_score(y_test, new_pred_prob)
            original_auc = roc_auc_score(y_test, original_pred_prob)
            
            new_pr_auc = average_precision_score(y_test, new_pred_prob)
            original_pr_auc = average_precision_score(y_test, original_pred_prob)
            
            comparison_results = {
                'new_model_auc': new_auc,
                'original_model_auc': original_auc,
                'auc_improvement': new_auc - original_auc,
                'new_model_pr_auc': new_pr_auc,
                'original_model_pr_auc': original_pr_auc,
                'pr_auc_improvement': new_pr_auc - original_pr_auc,
                'comparison_available': True
            }
            
            print(f"model comparison completed:")
            print(f"  new model auc: {new_auc:.3f}")
            print(f"  original model auc: {original_auc:.3f}")
            print(f"  improvement: {new_auc - original_auc:.3f}")
            
        except Exception as e:
            print(f"original model prediction failed: {e}")
            comparison_results['comparison_available'] = False
            
    except Exception as e:
        print(f"original model not available: {e}")
        comparison_results['comparison_available'] = False
    
    return comparison_results

def create_prediction_visualizations(results: pd.DataFrame) -> None:
    """create visualizations for recent rookie predictions"""
    print("creating prediction visualizations...")
    
    setup_plotting_style()
    
    if len(results) == 0:
        print("no results to visualize")
        return
    
    # prediction distribution by year
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # probability distribution by year
    ax1 = axes[0, 0]
    for year in sorted(results['rookie_year'].unique()):
        year_data = results[results['rookie_year'] == year]
        ax1.hist(year_data['probability'], bins=20, alpha=0.6, label=f'{int(year)}', density=True)
    
    ax1.set_xlabel('Success Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Probability Distribution by Rookie Year')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # top prospects
    ax2 = axes[0, 1]
    top_prospects = results.head(15)
    bars = ax2.barh(range(len(top_prospects)), top_prospects['probability'])
    ax2.set_yticks(range(len(top_prospects)))
    ax2.set_yticklabels(top_prospects['player_name'], fontsize=8)
    ax2.set_xlabel('Success Probability')
    ax2.set_title('Top 15 Prospects by Success Probability')
    ax2.invert_yaxis()
    
    # add probability labels
    for i, (bar, prob) in enumerate(zip(bars, top_prospects['probability'])):
        ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=8)
    
    # confidence intervals - fixed error calculation
    ax3 = axes[1, 0]
    top_20 = results.head(20)
    x_pos = range(len(top_20))
    
    # calculate error bars properly
    lower_errors = np.maximum(top_20['probability'] - top_20['ci_lower'], 0)
    upper_errors = np.maximum(top_20['ci_upper'] - top_20['probability'], 0)
    
    ax3.errorbar(x_pos, top_20['probability'], 
                yerr=[lower_errors, upper_errors], 
                fmt='o', capsize=3, capthick=1)
    ax3.set_xticks(x_pos[::2])
    ax3.set_xticklabels([name[:10] for name in top_20['player_name'].iloc[::2]], rotation=45)
    ax3.set_ylabel('Success Probability')
    ax3.set_title('Top 20 Prospects with Confidence Intervals')
    ax3.grid(True, alpha=0.3)
    
    # risk categories
    ax4 = axes[1, 1]
    risk_counts = results['risk_category'].value_counts()
    colors = ['lightcoral', 'orange', 'lightblue', 'lightgreen']
    wedges, texts, autotexts = ax4.pie(risk_counts.values, labels=risk_counts.index, 
                                      autopct='%1.1f%%', colors=colors)
    ax4.set_title('Distribution by Risk Category')
    
    plt.tight_layout()
    save_figure(fig, 'recent_rookie_predictions.png')
    
    # draft position vs prediction
    if 'draft_pick' in results.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # filter out undrafted players for cleaner visualization
        drafted = results[results['draft_pick'] > 0]
        
        if len(drafted) > 0:
            scatter = ax.scatter(drafted['draft_pick'], drafted['probability'], 
                               c=drafted['rookie_year'], cmap='viridis', 
                               s=60, alpha=0.7)
            
            # add trend line
            if len(drafted) > 5:
                z = np.polyfit(drafted['draft_pick'], drafted['probability'], 1)
                p = np.poly1d(z)
                ax.plot(drafted['draft_pick'], p(drafted['draft_pick']), 
                       "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Draft Pick')
            ax.set_ylabel('Success Probability')
            ax.set_title('Success Probability vs Draft Position (Recent Rookies)')
            ax.grid(True, alpha=0.3)
            
            # add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Rookie Year')
            
            # annotate top prospects
            top_5 = results.head(5)
            for _, row in top_5.iterrows():
                if row['draft_pick'] > 0:
                    ax.annotate(row['player_name'], 
                               (row['draft_pick'], row['probability']),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
        
        save_figure(fig, 'draft_vs_prediction.png')

def save_prediction_results(results: pd.DataFrame, comparison: Dict[str, Any]) -> None:
    """save prediction results and analysis"""
    print("saving prediction results...")
    
    # save detailed predictions
    predictions_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/recent_rookie_predictions.csv'
    results.to_csv(predictions_path, index=False)
    print(f"predictions saved to: {predictions_path}")
    
    # create summary report
    report_lines = []
    report_lines.append("# recent rookie predictions summary\n\n")
    
    if len(results) > 0:
        # overall statistics
        report_lines.append("## prediction summary\n\n")
        report_lines.append(f"- **total rookies analyzed**: {len(results)}\n")
        report_lines.append(f"- **years covered**: {', '.join(map(str, sorted(results['rookie_year'].unique())))}\n")
        report_lines.append(f"- **average success probability**: {results['probability'].mean():.3f}\n")
        report_lines.append(f"- **high-confidence predictions (>0.7)**: {len(results[results['probability'] > 0.7])}\n")
        
        # top prospects
        report_lines.append("\n## top 10 prospects\n\n")
        report_lines.append("| rank | player | year | team | draft | probability | confidence interval |\n")
        report_lines.append("|------|--------|------|------|-------|-------------|--------------------|\n")
        
        top_10 = results.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            draft_str = f"R{int(row['draft_round'])}-{int(row['draft_pick'])}" if row['draft_pick'] > 0 else "UDFA"
            
            report_lines.append(
                f"| {i} | {row['player_name']} | {int(row['rookie_year'])} | "
                f"{row['team']} | {draft_str} | {row['probability']:.3f} | {ci_str} |\n"
            )
        
        # risk category breakdown
        risk_dist = results['risk_category'].value_counts()
        report_lines.append("\n## risk category distribution\n\n")
        for category, count in risk_dist.items():
            pct = count / len(results) * 100
            report_lines.append(f"- **{category}**: {count} players ({pct:.1f}%)\n")
        
        # year-by-year analysis
        report_lines.append("\n## year-by-year analysis\n\n")
        for year in sorted(results['rookie_year'].unique()):
            year_data = results[results['rookie_year'] == year]
            avg_prob = year_data['probability'].mean()
            top_prospect = year_data.iloc[0]
            
            report_lines.append(f"### {int(year)} rookie class\n")
            report_lines.append(f"- **players analyzed**: {len(year_data)}\n")
            report_lines.append(f"- **average probability**: {avg_prob:.3f}\n")
            report_lines.append(f"- **top prospect**: {top_prospect['player_name']} ({top_prospect['probability']:.3f})\n\n")
    
    # model comparison
    if comparison.get('comparison_available', False):
        report_lines.append("\n## model comparison with original\n\n")
        report_lines.append(f"- **new model roc auc**: {comparison['new_model_auc']:.3f}\n")
        report_lines.append(f"- **original model roc auc**: {comparison['original_model_auc']:.3f}\n")
        report_lines.append(f"- **improvement**: {comparison['auc_improvement']:.3f}\n")
        report_lines.append(f"- **new model pr auc**: {comparison['new_model_pr_auc']:.3f}\n")
        report_lines.append(f"- **original model pr auc**: {comparison['original_model_pr_auc']:.3f}\n")
        report_lines.append(f"- **pr auc improvement**: {comparison['pr_auc_improvement']:.3f}\n")
    
    # save report
    report_path = '/home/yeblad/Desktop/New_WR_analysis/outputs/prediction_summary_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"prediction summary saved to: {report_path}")

def main():
    """main function to execute recent rookie prediction pipeline"""
    print("starting recent rookie prediction analysis")
    print("="*60)
    
    # load model and training data
    model, X_train, y_train = load_model_and_training_data()
    
    if model is None:
        print("no model available for predictions")
        return
    
    # load cleaned dataset to find recent rookies
    try:
        full_dataset = pd.read_parquet('/home/yeblad/Desktop/New_WR_analysis/outputs/cleaned_dataset.parquet')
    except:
        print("cleaned dataset not available")
        return
    
    # identify recent rookies
    recent_rookies = identify_recent_rookies(full_dataset)
    
    if len(recent_rookies) == 0:
        print("no recent rookies found")
        return
    
    # prepare features for recent rookies
    X_recent = prepare_recent_rookie_features(recent_rookies, X_train)
    
    # make predictions with confidence intervals
    results = make_predictions_with_confidence(model, X_recent, recent_rookies)
    
    if len(results) == 0:
        print("prediction failed")
        return
    
    # compare with original model
    comparison = compare_with_original_model()
    
    # create visualizations
    create_prediction_visualizations(results)
    
    # save results
    save_prediction_results(results, comparison)
    
    print("\nrecent rookie prediction analysis completed successfully!")
    print("results saved to /home/yeblad/Desktop/New_WR_analysis/outputs/")
    print("visualizations saved to /home/yeblad/Desktop/New_WR_analysis/figs/")

if __name__ == "__main__":
    main()
