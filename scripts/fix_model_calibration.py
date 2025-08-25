#!/usr/bin/env python3
"""
Analyze and fix model calibration issues
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figs"

def analyze_current_model():
    """Analyze why the current model is so conservative"""
    
    # Load the data and model
    X = pd.read_parquet(OUTPUT_DIR / 'features_X.parquet')
    y = pd.read_parquet(OUTPUT_DIR / 'target_y.parquet')['target']
    model = joblib.load(OUTPUT_DIR / 'best_model.pkl')
    
    print("=== CURRENT MODEL ANALYSIS ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Target rate: {y.mean():.3f}")
    
    # Get predictions on training data
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    print(f"\n=== PREDICTION DISTRIBUTION ===")
    print(f"Min probability: {y_pred_proba.min():.3f}")
    print(f"Max probability: {y_pred_proba.max():.3f}")
    print(f"Mean probability: {y_pred_proba.mean():.3f}")
    print(f"Median probability: {np.median(y_pred_proba):.3f}")
    
    # Check how many predictions are above various thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n=== PREDICTIONS BY THRESHOLD ===")
    for thresh in thresholds:
        count = (y_pred_proba >= thresh).sum()
        pct = count / len(y_pred_proba) * 100
        print(f"Predictions >= {thresh:.1f}: {count} ({pct:.1f}%)")
    
    # Analyze true positives
    true_positives = X[y == 1]
    tp_predictions = model.predict_proba(true_positives)[:, 1]
    
    print(f"\n=== TRUE POSITIVE ANALYSIS ===")
    print(f"Number of actual 1000-yard achievers: {len(true_positives)}")
    print(f"Their average predicted probability: {tp_predictions.mean():.3f}")
    print(f"Max probability for true positive: {tp_predictions.max():.3f}")
    print(f"How many predicted > 0.5: {(tp_predictions > 0.5).sum()}")
    print(f"How many predicted > 0.3: {(tp_predictions > 0.3).sum()}")
    
    # Look at feature scales
    print(f"\n=== FEATURE SCALE ANALYSIS ===")
    print("Top 5 features by variance:")
    variances = X.var().sort_values(ascending=False)
    print(variances.head())
    
    print("\nTop 5 features by mean:")
    means = X.mean().sort_values(ascending=False)
    print(means.head())
    
    return model, X, y, y_pred_proba

def fix_probability_calibration(model, X, y):
    """Apply probability calibration to make predictions less conservative"""
    
    print("\n=== APPLYING CALIBRATION ===")
    
    # Split data for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try different calibration methods
    calibration_methods = ['sigmoid', 'isotonic']
    
    best_calibrated = None
    best_score = -1
    
    for method in calibration_methods:
        print(f"\nTrying {method} calibration...")
        
        # Create calibrated classifier
        calibrated = CalibratedClassifierCV(
            model, 
            method=method, 
            cv='prefit'  # Since model is already trained
        )
        
        # Fit calibration on held-out data
        calibrated.fit(X_cal, y_cal)
        
        # Check new predictions
        y_cal_pred = calibrated.predict_proba(X_cal)[:, 1]
        
        print(f"Calibrated predictions - Min: {y_cal_pred.min():.3f}, Max: {y_cal_pred.max():.3f}")
        print(f"Mean: {y_cal_pred.mean():.3f}, Std: {y_cal_pred.std():.3f}")
        
        # Check distribution
        high_conf = (y_cal_pred > 0.5).sum()
        print(f"High confidence predictions (>0.5): {high_conf}")
        
        if y_cal_pred.max() > best_score:
            best_score = y_cal_pred.max()
            best_calibrated = calibrated
    
    return best_calibrated

def adjust_class_weights(X, y):
    """Retrain with adjusted class weights to be less conservative"""
    
    print("\n=== ADJUSTING CLASS WEIGHTS ===")
    
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.preprocessing import StandardScaler
    
    # Create different weight configurations
    weight_configs = [
        {'scale_pos_weight': 1},  # Balanced
        {'scale_pos_weight': 2},  # Moderate boost
        {'scale_pos_weight': 5},  # Strong boost
        {'scale_pos_weight': 10}, # Very strong boost
    ]
    
    best_model = None
    best_range = 0
    
    for config in weight_configs:
        print(f"\nTesting scale_pos_weight = {config['scale_pos_weight']}")
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', SMOTE(random_state=42)),
            ('model', XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                **config
            ))
        ])
        
        # Train model
        pipeline.fit(X, y)
        
        # Check predictions
        y_pred_proba = pipeline.predict_proba(X)[:, 1]
        
        prob_range = y_pred_proba.max() - y_pred_proba.min()
        print(f"Probability range: {y_pred_proba.min():.3f} to {y_pred_proba.max():.3f}")
        print(f"Predictions > 0.5: {(y_pred_proba > 0.5).sum()}")
        print(f"Mean probability: {y_pred_proba.mean():.3f}")
        
        if prob_range > best_range:
            best_range = prob_range
            best_model = pipeline
    
    return best_model

def main():
    """Main execution"""
    
    print("="*60)
    print("MODEL CALIBRATION ANALYSIS AND FIXES")
    print("="*60)
    
    # Analyze current model
    model, X, y, y_pred_proba = analyze_current_model()
    
    # Create visualization of current predictions
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Current Model - Prediction Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='0.5 threshold')
    plt.legend()
    
    # Try calibration
    calibrated_model = fix_probability_calibration(model, X, y)
    
    if calibrated_model:
        y_cal_pred = calibrated_model.predict_proba(X)[:, 1]
        
        plt.subplot(1, 3, 2)
        plt.hist(y_cal_pred, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('After Calibration')
        plt.axvline(x=0.5, color='red', linestyle='--', label='0.5 threshold')
        plt.legend()
    
    # Try adjusted class weights
    adjusted_model = adjust_class_weights(X, y)
    
    if adjusted_model:
        y_adj_pred = adjusted_model.predict_proba(X)[:, 1]
        
        plt.subplot(1, 3, 3)
        plt.hist(y_adj_pred, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('With Adjusted Class Weights')
        plt.axvline(x=0.5, color='red', linestyle='--', label='0.5 threshold')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'calibration_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {FIG_DIR / 'calibration_comparison.png'}")
    
    # Save the best model
    if adjusted_model:
        joblib.dump(adjusted_model, OUTPUT_DIR / 'model_calibrated.pkl')
        print(f"\nCalibrated model saved to {OUTPUT_DIR / 'model_calibrated.pkl'}")
    
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
