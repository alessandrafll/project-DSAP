# ESG Risk Prediction Project

This repository contains my **individual final project** for the course  
**Advanced Programming: Data Science (Fall 2025)** at HEC Lausanne.

The objective of this project is to analyze and predict **ESG Risk Levels** of S&P 500 companies using publicly available ESG data and financial indicators, and to compare the performance of two machine learning models.

---

## Project Overview

Environmental, Social, and Governance (ESG) risk has become a central concern for investors and policymakers.  
This project investigates whether basic firm-level financial characteristics can help predict ESG risk categories.

The project combines:
- ESG risk data from Kaggle (S&P 500 ESG Risk Ratings),
- Financial indicators retrieved from Yahoo Finance,
- Supervised machine learning models to predict ESG risk levels.

In addition to model comparison, a sector-level analysis is performed to explore differences in ESG risk across industries.

---

## Research Question

**Can ESG Risk Levels be predicted using firm-level financial indicators, and which model performs best: Logistic Regression or Random Forest?**

---

## Repository Structure
 - README.md 
 - PROPOSAL.md 
 - AI_Usage.md
 - requirements.txt
 - main.py
 - src/ 
   - __init__.py
   - data_loader.py
   - models.py
   - evaluation.py
 - data/ 
   - raw/
   - processed/
 - notebooks/
 - results/ 
 

---

## Data

### ESG dataset Kaggle

- Dataset: https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings
- Source: Kaggle (originally based on Sustainalytics ESG data)

### How to obtain the data

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings
2. Extract the CSV file.
3. Rename it to: esg_risk_ratings.csv
4. Place the file in the following directory: data/raw/esg_risk_ratings.csv

*The project will raise an explicit error if the file is missing.*

## Financial Data (Yahoo Finance)

Additional financial variables are retrieved using the `yfinance` library:
- Market Capitalization
- P/E Ratio
- Return on Equity (ROE)
- Debt-to-Equity Ratio
- Beta
- Dividend Yield

To ensure reproducibility, Yahoo Finance data are cached locally in: data/processed/yf_cache.csv
This file is created automatically on the first run of the project.

On subsequent runs, the cached file is reused instead of querying Yahoo Finance again.

---

## Models

Two classification models are implemented and compared:

1. **Logistic Regression**
   - Baseline linear classifier
   - Features standardized using `StandardScaler`

2. **Random Forest Classifier**
   - Non-linear ensemble model
   - Provides feature importance measures

---

## Evaluation

Models are evaluated on a held-out test set using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrices (saved as PNG images)

All results are automatically saved in the `results/` directory:
- Per-model metrics (`.json`)
- Summary comparison table (`.csv`)
- Confusion matrix figures (`.png`)
- Feature importance plot for Random Forest (`.png`)

---

## Sector-Level Analysis (Stretch Goal)

As an additional analysis, the project computes:
- The **average ESG Risk Score by sector**
- A bar plot comparing ESG risk across sectors

Outputs:
- `results/sector_esg_risk_summary.csv`
- `results/sector_esg_risk_comparison.png`

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Run the pipeline
```bash
python main.py
```

This command:
- Builds the dataset
- Trains both models
- Evaluates performance
- Saves all results and figures in results/

## Reproducibility

- Fixed random seeds are used (random_state = 42)
- Yahoo Finance data are cached locally
- Running python main.py multiple times produces identical results (given the same data files)


## Author
**Alessandra Failla**  
HEC Lausanne - University of Lausanne
Fall 2025

## AI Tools Usage
AI tools (ChatGPT) were used as learning and debugging assistance, in accordance with the course policy.
All code was reviewed, understood, and adapted by the author.
Details are provided in AI_USAGE.md.
