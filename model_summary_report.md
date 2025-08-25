# model evaluation summary report

## best performing model: xgboost

### performance metrics
- **Roc Auc**: 0.9784
- **Pr Auc**: 0.9072
- **F1**: 0.7921
- **Recall**: 0.9302
- **Precision**: 0.6897
- **Brier Score**: 0.0550

### best hyperparameters
- **model__subsample**: 1.0
- **model__scale_pos_weight**: 3
- **model__n_estimators**: 300
- **model__min_child_weight**: 5
- **model__max_depth**: 5
- **model__learning_rate**: 0.01
- **model__gamma**: 0.1
- **model__colsample_bytree**: 0.8

## model comparison

| model | roc auc | pr auc | f1 score | recall | precision |
|-------|---------|--------|----------|--------|-----------|
| logistic_regression | 0.977 | 0.853 | 0.642 | 0.988 | 0.475 |
| random_forest | 0.970 | 0.758 | 0.816 | 0.930 | 0.727 |
| gradient_boosting | 0.971 | 0.869 | 0.775 | 0.802 | 0.750 |
| xgboost | 0.978 | 0.907 | 0.792 | 0.930 | 0.690 |

## cross-validation stability

| model | roc auc (std) | pr auc (std) | f1 (std) |
|-------|---------------|--------------|----------|
| logistic_regression | 0.978 (±0.007) | 0.864 (±0.053) | 0.647 (±0.062) |
| random_forest | 0.971 (±0.013) | 0.785 (±0.099) | 0.819 (±0.060) |
| gradient_boosting | 0.971 (±0.018) | 0.882 (±0.056) | 0.776 (±0.065) |
| xgboost | 0.978 (±0.011) | 0.903 (±0.038) | 0.796 (±0.066) |

## key insights

- **best model**: xgboost with ROC AUC of 0.978
- **worst model**: random_forest with ROC AUC of 0.970
- **performance gap**: 0.009 ROC AUC points
- **most stable model**: logistic_regression with ROC AUC std of 0.007
- **precision lift over baseline**: 5.12x
