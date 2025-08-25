# Improved Model Report
**Date**: 2025-08-24 21:36

## Model Improvements
- ✅ Removed 'rec' feature (correlation 0.78 with target)
- ✅ Removed 26 highly correlated/low variance features
- ✅ Reduced model complexity (shallower trees, fewer estimators)
- ✅ Applied stronger regularization (L1/L2)
- ✅ Used temporal validation instead of random splits
- ✅ Applied probability calibration

## Performance Metrics
- **ROC AUC**: 0.947
- **Average Precision**: 0.753
- **Brier Score**: 0.073 (lower is better)

## Key Features Used
Total features: 20

Top features:
1. draft_round
2. age
3. rec_td
4. yards_per_reception
5. draft_capital_score
6. early_round
7. first_round
8. top_3_wr
9. top_5_wr
10. modern_era

## Expected Improvements
- Better generalization to future rookies
- More realistic probability estimates
- Reduced overfitting on training data
- More interpretable predictions
