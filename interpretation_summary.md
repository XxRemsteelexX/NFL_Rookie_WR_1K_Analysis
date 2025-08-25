# model interpretation summary

## top 20 most important features

| rank | feature | importance | description |
|------|---------|------------|-------------|
| 1 | rec | 0.2066 | total receptions in rookie season |
| 2 | recent_era | 0.0047 | engineered feature |
| 3 | wr_draft_rank | 0.0047 | engineered feature |
| 4 | target_percentile | 0.0044 | engineered feature |
| 5 | age | 0.0022 | age during rookie season |
| 6 | yards_percentile | 0.0022 | engineered feature |
| 7 | rec_yards_zscore | 0.0022 | composite score metric |
| 8 | td_per_reception | 0.0019 | engineered feature |
| 9 | draft_round | 0.0019 | draft round (1-7) |
| 10 | modern_era_x_yards | 0.0016 | interaction feature |
| 11 | day_1_pick | 0.0013 | engineered feature |
| 12 | first_round | 0.0013 | engineered feature |
| 13 | draft_pick | 0.0013 | overall draft position |
| 14 | early_round | 0.0009 | engineered feature |
| 15 | rec_50_plus | 0.0003 | binary indicator for achieving threshold |
| 16 | yards_per_target | 0.0000 | engineered feature |
| 17 | yards_per_reception | 0.0000 | average yards per reception |
| 18 | targets | 0.0000 | total targets in rookie season |
| 19 | catch_rate | 0.0000 | receptions divided by targets |
| 20 | rec_td | 0.0000 | receiving touchdowns in rookie season |

## key insights

- **most important feature**: rec with importance 0.2066
- **draft-related features in top 20**: 3
- **performance features in top 20**: 11
- **efficiency features in top 20**: 6
