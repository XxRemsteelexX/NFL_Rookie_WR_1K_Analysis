# feature engineering documentation

## feature categories and descriptions

### Base Stats
- **rec**: total receptions in rookie season
- **rec_yards**: total receiving yards in rookie season
- **rec_td**: total receiving touchdowns in rookie season
- **targets**: total targets in rookie season
- **age**: engineered feature

### Draft Info
- **draft_round**: engineered feature
- **draft_pick**: engineered feature
- **draft_capital_score**: normalized draft position score (higher = earlier pick)
- **early_round**: drafted in rounds 1-2 (binary)
- **first_round**: engineered feature

### Efficiency
- **catch_rate**: receptions divided by targets
- **yards_per_reception**: receiving yards divided by receptions
- **yards_per_target**: receiving yards divided by targets
- **target_rate**: engineered feature

### Thresholds
- **yards_500_plus**: achieved 500+ yards in rookie season (binary)
- **rec_50_plus**: achieved 50+ rec in rookie season (binary)
- **targets_80_plus**: achieved 80+ targets in rookie season (binary)
- **td_3_plus**: achieved 3+ td in rookie season (binary)
- **catch_rate_65_plus**: achieved 65+ catch rate in rookie season (binary)
- **ypr_12_plus**: achieved 12+ ypr in rookie season (binary)

### Interactions
- **draft_x_yards**: interaction between draft and yards
- **early_round_x_targets**: interaction between early round and targets
- **volume_x_efficiency**: interaction between volume and efficiency
- **modern_era_x_yards**: interaction between modern era and yards

### Composites
- **draft_capital_score**: normalized draft position score (higher = earlier pick)
- **rookie_production_score**: weighted composite of rookie production metrics
- **efficiency_score**: composite efficiency rating
- **breakout_score**: probability of future breakout based on rookie indicators
- **rec_yards_zscore**: z-score normalized rec yards within rookie class

### Transformations
- **rec_yards_log**: log transformation of rec yards
- **targets_log**: log transformation of targets
- **rec_yards_sqrt**: square root transformation of rec yards
- **rec_yards_zscore**: z-score normalized rec yards within rookie class

## feature statistics

- **total features**: 45
- **numeric features**: 45
- **categorical features**: 0
