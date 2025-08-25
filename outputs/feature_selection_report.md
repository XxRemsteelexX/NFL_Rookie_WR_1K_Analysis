# Feature Analysis and Selection Report
**Date**: 2025-08-24 21:17
**Original Features**: 46
**Optimized Features**: 20
**Features Removed**: 26

## Removed Features
- **opportunity_x_efficiency**: near-zero variance
- **td_per_reception**: near-zero variance
- **target_rate**: near-zero variance
- **td_3_plus**: near-zero variance
- **rookie_production_score**: near-zero variance
- **efficiency_score**: near-zero variance
- **high_target_rookie**: near-zero variance
- **early_round_x_targets**: high feature correlation
- **yards_500_plus**: high feature correlation
- **yards_per_target**: high feature correlation
- **rec_yards_sqrt**: high feature correlation
- **draft_pick**: high feature correlation
- **yards_per_route_run**: high feature correlation
- **targets_log**: high feature correlation
- **targets**: high feature correlation
- **catch_rate**: high feature correlation
- **wr_draft_rank**: high feature correlation
- **rec_yards**: high feature correlation
- **volume_x_efficiency**: high feature correlation
- **rec**: target correlation 0.780
- **targets_80_plus**: low mutual information
- **yac_per_reception**: low mutual information
- **recent_era**: low mutual information
- **day_1_pick**: low mutual information
- **modern_era_x_yards**: low mutual information
- **ypr_12_plus**: low mutual information

## Performance Comparison
| Feature Set | CV ROC AUC | Temporal ROC AUC | Gap |
|------------|------------|------------------|-----|
| all_features | 0.954 | 0.913 | 0.041 |
| no_rec | 0.955 | 0.890 | 0.064 |
| basic_only | 0.735 | 0.735 | 0.000 |
| no_high_corr | 0.945 | 0.858 | 0.088 |
| top_mi | 0.952 | 0.891 | 0.061 |
| no_dominant | 0.937 | 0.888 | 0.049 |
| rfe_selected | 0.963 | N/A | N/A |
