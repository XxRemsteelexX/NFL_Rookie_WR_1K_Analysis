# model evaluation summary report

## best performing model: xgboost

### performance metrics
- **Roc Auc**: 0.9791
- **Pr Auc**: 0.9029
- **F1**: 0.7921
- **Recall**: 0.9302
- **Precision**: 0.6897
- **Brier Score**: 0.0553

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
| random_forest | 0.973 | 0.794 | 0.814 | 0.919 | 0.731 |
| gradient_boosting | 0.972 | 0.876 | 0.780 | 0.826 | 0.740 |
| xgboost | 0.979 | 0.903 | 0.792 | 0.930 | 0.690 |

## cross-validation stability

| model | roc auc (std) | pr auc (std) | f1 (std) |
|-------|---------------|--------------|----------|
| logistic_regression | 0.978 (+/-0.007) | 0.864 (+/-0.053) | 0.647 (+/-0.062) |
| random_forest | 0.972 (+/-0.011) | 0.806 (+/-0.089) | 0.817 (+/-0.055) |
| gradient_boosting | 0.970 (+/-0.018) | 0.869 (+/-0.055) | 0.782 (+/-0.077) |
| xgboost | 0.979 (+/-0.009) | 0.902 (+/-0.033) | 0.796 (+/-0.066) |

## key insights

- **best model**: xgboost with ROC AUC of 0.979
- **worst model**: gradient_boosting with ROC AUC of 0.972
- **performance gap**: 0.007 ROC AUC points
- **most stable model**: logistic_regression with ROC AUC std of 0.007
- **precision lift over baseline**: 5.12x
