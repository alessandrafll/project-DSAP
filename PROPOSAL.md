# Project Proposal: Predicting ESG Risk Levels Using Public Data

**Category:** Data Science / Machine Learning  

---

## Problem Statement and Motivation

Environmental, Social, and Governance (ESG) scores are increasingly used to assess corporate sustainability and risk. However, most ESG datasets are proprietary and costly, making them inaccessible to students and small-scale researchers.  

This project aims to explore whether ESG risk levels can be predicted using publicly available financial data by applying and comparing machine learning methods.  
The focus is on the application of data science models to understand whether basic financial indicators can explain variations in ESG risk among large U.S. companies.  

---

## Data Sources and Structure

- **Primary ESG Data:**  
  The *S&P 500 ESG Risk Ratings* dataset from [Kaggle](https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings) provides a cross-sectional snapshot for 2023.  
  The data is based on Sustainalytics and S&P Global ESG Risk Ratings, which are widely recognized benchmarks for measuring corporate sustainability.  
  Each company in the dataset is assigned an overall ESG risk score (lower = better sustainability).

- **Financial and Company Data:**  
  Firm-level financial indicators will be retrieved from Yahoo Finance using the `yfinance` Python library.  
  The following variables will be matched to the same time period (2023):  
  - Sector and Industry  
  - Market Capitalization  
  - Price-to-Earnings (P/E) Ratio  
  - Return on Equity (ROE)  
  - Debt-to-Equity Ratio  
  - Beta (volatility measure)  
  - Dividend Yield  

Together, these datasets form a 2023 cross-section of large-cap U.S. firms suitable for predictive modeling.

---

## Planned Approach and Technologies

1. **Data Preparation**
   - Import and clean both datasets (Kaggle ESG and Yahoo Finance).  
   - Merge them by company ticker, handle missing values, and normalize numeric features.  
   - Ensure that all data reflect the same period (cross-section for 2023).  

2. **Machine Learning Application**
   - Transform ESG risk scores into three categories: *Low*, *Medium*, and *High* risk.  
   - Apply and compare two supervised models:  
     - **Logistic Regression** (interpretable baseline)  
     - **Random Forest Classifier** (non-linear benchmark)  
   - Evaluate performance using cross-validation, accuracy, F1-score, and a confusion matrix.  
   - Perform a brief model comparison to assess which method best predicts ESG levels.  

3. **Visualization**
   - Display relationships between financial indicators and ESG levels.  
   - Include feature importance charts and performance comparison plots.  

---

## Expected Challenges and Solutions
- **Missing or incomplete financial data:** fill with sector averages or remove affected cases.  
- **Imbalanced ESG classes:** use class weighting or resampling.  
- **Data alignment issues:** standardize tickers and ensure consistency across sources.  
- **Reproducibility:** maintain a documented and clean Jupyter Notebook with fixed random seeds.

---

## Success Criteria
- A merged dataset with at least 50–100 S&P 500 companies (cross-section for 2023).  
- Two machine learning models successfully implemented and compared.  
- Clear quantitative results and visualizations supporting model evaluation.  
- Fully reproducible workflow using only public data sources.

---

## Stretch Goals (Optional) 
- Analyze differences in ESG prediction across sectors (e.g., Technology vs Energy).  
- Build a simple Streamlit dashboard for interactive results exploration.
