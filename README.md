# NFL Rookie Wide Receiver 1000+ Yard Season Prediction

## Advanced Machine Learning Analysis with Feature Optimization and Temporal Validation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Project Overview

This project develops a machine learning model to predict which NFL rookie wide receivers will achieve at least one 1000+ yard receiving season in their career. The analysis includes comprehensive feature engineering, overfitting diagnosis and remediation, and temporal validation to ensure robust predictions on future rookies.

### Key Results

- **Initial Model**: 97.9% ROC AUC (with severe overfitting)
- **Improved Model**: 90.9% ROC AUC on future data (minimal overfitting)
- **Overfitting Reduction**: From 18.5% to 0.4% gap
- **Feature Reduction**: From 46 to 20 features
- **Business Impact**: Production-ready model for draft analysis

## Repository Structure

```
NFL_Rookie_WR_1K_Analysis/
│
├── Downloads/                      # Raw data files
│   ├── 2006-2024 rookie data     # Rookie season statistics
│   ├── rec2006-2023.csv          # Reception data by year
│   └── 1kseasons.csv             # Career outcome data
│
├── Uploads/                        # Original uploaded files
│   ├── rookie data files
│   ├── advanced metrics files
│   └── model.pkl
│
├── outputs/                        # Processed data and models
│   ├── cleaned_dataset.parquet    # Integrated dataset
│   ├── features_X.parquet         # Original features
│   ├── features_X_optimized.parquet # Optimized features
│   ├── target_y.parquet           # Target variable
│   ├── model_metrics.csv          # Model performance metrics
│   ├── improved_model.pkl         # Final improved model
│   ├── improved_model_report.json # Final model report
│   └── feature_selection_report.md # Feature analysis report
│
├── figs/                           # Visualizations
│   ├── draft_analysis.png         # Draft round analysis
│   ├── feature_correlation_matrix.png
│   ├── overfitting_analysis.png
│   ├── improved_model_evaluation.png
│   ├── calibration_comparison.png
│   └── feature_selection_analysis.png
│
├── notebooks/
│   └── NFL_WR_Analysis_Complete.ipynb  # Comprehensive analysis notebook
│
├── scripts/                        # Analysis scripts
│   ├── 01_build_dataset.py       # Data loading and merging
│   ├── 02_eda.py                 # Exploratory data analysis
│   ├── 03_feature_eng.py         # Feature engineering
│   ├── 04_modeling.py            # Model training
│   ├── 05_interpret.py           # Model interpretation
│   ├── 06_predict_recent.py      # Recent rookie predictions
│   ├── 07_feature_analysis_selection.py  # Feature optimization
│   ├── 08_improved_model.py      # Improved model with temporal validation
│   ├── build_notebook.py         # Notebook generation
│   ├── build_final_notebook.py   # Final notebook creation
│   ├── fix_data_leakage.py       # Data leakage fixes
│   ├── fix_model_calibration.py  # Model calibration
│   └── utils.py                  # Utility functions
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── utils.py
│   └── visuals.py
│
├── Old Files/                     # Legacy HTML reports
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## Quick Start

### Prerequisites

- Python 3.10+
- Anaconda or Miniconda (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/XxRemsteelexX/NFL_Rookie_WR_1K_Analysis.git
cd NFL_Rookie_WR_1K_Analysis
```

2. Create conda environment:
```bash
conda create -n wr_analysis python=3.10
conda activate wr_analysis
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Run the Complete Pipeline
```bash
# Run all scripts in sequence
python scripts/01_build_dataset.py
python scripts/02_eda.py
python scripts/03_feature_eng.py
python scripts/04_modeling.py
python scripts/05_interpret.py
python scripts/06_predict_recent.py
python scripts/07_feature_analysis_selection.py
python scripts/08_improved_model.py
```

#### Option 2: Use the Jupyter Notebook
```bash
jupyter notebook notebooks/NFL_WR_Analysis_Complete.ipynb
```

## Key Findings

### 1. The Overfitting Problem

The original model achieved exceptional cross-validation performance (97.9% ROC AUC) but suffered from severe overfitting when tested on future years:

| Validation Type | Train ROC AUC | Test ROC AUC | Overfitting Gap |
|-----------------|---------------|--------------|-----------------|
| Cross-Validation | 1.000 | 0.979 | 0.021 |
| Temporal (2018+) | 1.000 | 0.931 | 0.069 |
| Temporal (2020+) | 1.000 | 0.885 | 0.115 |
| Temporal (2021+) | 1.000 | 0.815 | 0.185 |

### 2. Root Cause Analysis

- **Dominant Feature**: The 'rec' (receptions) feature had 0.78 correlation with the target
- **Multicollinearity**: 39 feature pairs with correlation > 0.85
- **Feature Bloat**: 46 features leading to model memorization

### 3. Solution Implementation

- Removed problematic features (especially 'rec')
- Reduced feature set from 46 to 20
- Applied stronger regularization
- Implemented proper temporal validation
- Added probability calibration

### 4. Improved Results

| Metric | Original Model | Improved Model | Improvement |
|--------|---------------|----------------|-------------|
| Features | 46 | 20 | -56.5% |
| ROC AUC (CV) | 0.979 | 0.947 | -3.3% |
| ROC AUC (Temporal) | 0.815 | 0.909 | +11.5% |
| Overfitting Gap | 0.185 | 0.004 | -97.8% |
| Interpretability | Low | High | Improved |

## Methodology

### Data Processing
1. **Integration**: Merged draft, rookie, and career data from 2006-2024
2. **Cleaning**: Handled missing values, removed duplicates
3. **Target Creation**: Binary classification (1000+ yard season achieved)

### Feature Engineering
- Basic statistics (yards, touchdowns, games)
- Efficiency metrics (yards per reception, catch rate)
- Draft capital features
- Production thresholds
- Composite scores

### Model Development
- Logistic Regression with L1/L2 regularization
- Random Forest with constrained depth
- XGBoost with early stopping
- Ensemble voting classifier

### Validation Strategy
- Stratified K-fold cross-validation
- **Temporal validation** (train on past, test on future)
- Probability calibration
- Feature importance analysis

## Visualizations

The project includes comprehensive visualizations:
- Draft round success rate analysis
- Feature correlation matrices
- Overfitting gap comparisons
- ROC curves and PR curves
- Feature importance plots
- Calibration plots

## Use Cases

1. **NFL Teams**: Draft evaluation and rookie selection
2. **Fantasy Football**: Player value predictions
3. **Sports Analytics**: Career trajectory modeling
4. **ML Education**: Case study in overfitting diagnosis

## Future Enhancements

- [ ] Incorporate college statistics
- [ ] Add NFL combine metrics
- [ ] Include team offensive system features
- [ ] Develop position-specific models (RB, TE)
- [ ] Create web application for predictions
- [ ] Implement real-time updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Project Link: [https://github.com/XxRemsteelexX/NFL_Rookie_WR_1K_Analysis](https://github.com/XxRemsteelexX/NFL_Rookie_WR_1K_Analysis)

## Acknowledgments

- NFL data sources
- scikit-learn and XGBoost communities
- Sports analytics community

---

**Last Updated**: August 2024
