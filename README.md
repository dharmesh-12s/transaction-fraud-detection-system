# 🎬 Transaction Fraud Detection System

A practical tutorial on building a machine learning system to detect fraudulent credit card transactions.

**[📊 View Full Interactive Tutorial](https://data-v-labs.github.io/transaction-fraud-detection-system/)**

---

## Overview

This project walks through the complete data science pipeline for fraud detection, from raw data to a trained predictive model. Using real transaction data from IEEE-CIS Kaggle dataset, we demonstrate how to identify fraudulent patterns and build an effective classifier.

---

## What We Found (Key EDA Findings)

### Class Imbalance
- **96.5%** of transactions are legitimate
- **3.5%** of transactions are fraudulent
- This heavy imbalance shapes our modeling approach

### Transaction Amount Patterns
- Fraudulent transactions have **lower median values** than legitimate ones
- Fraud detection cannot rely on high-value transactions as primary indicator
- Both classes show similar log-normal distribution patterns
- **99% of all transactions** fall in low-to-mid amount ranges

### Temporal Behavior
- Fraudsters show distinct patterns across hours and days
- Certain hours have elevated fraud rates compared to overall average
- Fraud activity trends vary significantly by time of day

### Feature Correlations
- **V45, V86, V87, V44, V52** show strongest correlation with fraud
- No single feature is highly predictive alone (no >0.8 correlation)
- Fraud is likely explained by **interaction of many features**

### Statistical Evidence
- **T-test p-value: 3.84 × 10⁻¹⁹** (highly significant)
- Mean transaction amounts differ significantly between fraud and non-fraud
- Thousands of outliers in both classes need careful handling

---

## Models Used

We tested and compared the following machine learning classifiers:

1. **Logistic Regression** - Baseline model for interpretability
2. **Random Forest Classifier** - Ensemble learning with feature importance
3. **XGBoost** - The champion model delivering best performance

**Winner: XGBoost** provided the best ROC-AUC score and fraud detection capability.

---

## Data Processing Steps

1. **Data Loading** - Combined transaction and identity tables
2. **Data Cleaning**
   - Handled missing values with iterative imputation
   - Removed duplicates
   - Fixed data type inconsistencies
3. **Feature Engineering** - Created meaningful features for modeling
4. **Train-Test Split** - Preserved fraud distribution in both sets

---

## Dataset

- **Source:** IEEE-CIS Fraud Detection (Kaggle)
- **Transaction Table:** 590,540 transactions with 30+ features
- **Identity Table:** Device and network information
- **Target:** isFraud (0 = legitimate, 1 = fraudulent)
- **Key Features:** Transaction amount, card details, email domains, temporal data, device info

---

## Key Technologies

- **Python** - Data processing and modeling
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Logistic Regression & Random Forest
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Statistical testing

---

## Project Structure

```
transaction-fraud-detection-system/
├── assets/
│   └── css/                                  # Styling for GitHub Pages
├── index_files/                              # Supporting files for hosted site
├── Transaction_fraud_detection_system.ipynb  # Complete analysis notebook (Colab)
├── _config.yml                               # Site configuration and metadata
├── index.md                                  # Main tutorial page
└── README.md                                 # This file
```

### Key Files

- **Transaction_fraud_detection_system.ipynb** - Full Jupyter notebook with all analysis, EDA, and model training code
- **index.md** - Main tutorial content hosted on GitHub Pages
- **_config.yml** - Jekyll configuration for the GitHub Pages site
- **assets/css/style.css** - CSS styling for the hosted website

---

## Questions Answered

✓ What percentage of transactions are fraudulent?  
✓ Are fraudsters spending more or less than legitimate customers?  
✓ What features correlate most with fraud?  
✓ Which ML models detect fraud most effectively?  
✓ How do temporal patterns reveal fraudulent activity?  

---

## Next Steps

- Deploy XGBoost model to production
- Implement real-time scoring API
- Monitor model performance over time
- Retrain with new fraud patterns as they emerge

---

## Authors

**Tanishqa Aggarwal and Shreevatsa Agnihotri**

---

**Ready to learn?** [Start the tutorial now](https://data-v-labs.github.io/transaction-fraud-detection-system/)
