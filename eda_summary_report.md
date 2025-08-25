# exploratory data analysis summary report

## dataset overview
- **total records:** 639
- **total features:** 100
- **numeric features:** 91
- **categorical features:** 9

## target variable analysis
- **overall success rate:** 13.5%
- **successful players:** 86
- **unsuccessful players:** 553
- **class imbalance ratio:** 6.4:1

## draft position analysis
| round | total | successful | success rate |
|-------|-------|------------|-------------|
| 0.0 | 30.0 | 0.0 | 0.0% |
| 1.0 | 75.0 | 32.0 | 42.7% |
| 2.0 | 87.0 | 24.0 | 27.6% |
| 3.0 | 94.0 | 15.0 | 16.0% |
| 4.0 | 84.0 | 3.0 | 3.6% |
| 5.0 | 71.0 | 6.0 | 8.5% |
| 6.0 | 99.0 | 4.0 | 4.0% |
| 7.0 | 99.0 | 2.0 | 2.0% |

## rookie performance comparison
| metric | successful avg | unsuccessful avg | difference |
|--------|----------------|------------------|------------|
| rec | 469.12 | 57.56 | 411.55 |
| rec yards | 47.34 | 34.62 | 12.72 |
| rec td | 0.15 | 0.03 | 0.12 |
| targets | 3.92 | 3.95 | -0.03 |
| catch rate | 0.02 | 0.06 | -0.04 |

## high missing data features (>50%)
- **drop_rate_rec:** 100.0% missing
- **declined_penalties:** 100.0% missing
- **contested_targets:** 100.0% missing
- **contested_receptions:** 100.0% missing
- **contested_catch_pct_rec:** 100.0% missing
- **caught_percent_rec:** 100.0% missing
- **avoided_tackles_rec:** 100.0% missing
- **avg_depth_of_target_rec:** 100.0% missing
- **player_game_count:** 100.0% missing
- **team_rec:** 100.0% missing

## key insights
- early round picks (1-2) have 29.2% success rate
- late round picks (5+) have 4.5% success rate
- rookies with 500+ yards have 15.8% success rate
- rookies with <200 yards have 13.7% success rate
