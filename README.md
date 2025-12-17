# ESG Risk Prediction Project

This repository contains my project for the **Data Science and Advanced programming** course.  
The goal is to explore whether ESG risk levels can be predicted using publicly available financial data.

---

## Project Overview
This project combines ESG scores from the *S&P 500 ESG Risk Ratings (2023)* dataset on Kaggle  
with company-level financial indicators from Yahoo Finance.  
The aim is to test whether basic firm characteristics (e.g., market cap, P/E ratio, ROE) can predict ESG risk categories.

---

## Repository Structure
 - README.md # Project documentation (this file)
 - PROPOSAL.md # Project proposal (idea and plan)
 - requirements.txt # Python libraries used (to be added later)
 - src/ # Source code (to be created later)
 - data/ # Sample or processed data (optional)
 - results/ # Output files or figures (later)
 - docs/ # Extra documentation (optional)


---

## Technologies
- Python 3
- Libraries: `pandas`, `numpy`, `yfinance`, `scikit-learn`, `matplotlib`, `seaborn`
- Notebook environment: Jupyter

---

## Planned Workflow
1. Load and clean the Kaggle ESG dataset (cross-section for 2023).  
2. Collect financial variables from Yahoo Finance using `yfinance`.  
3. Merge datasets and preprocess numeric features.  
4. Train and compare two models:
   - Logistic Regression (baseline)
   - Random Forest Classifier (non-linear benchmark)
5. Evaluate with cross-validation, F1-score, and confusion matrix.
6. Visualize relationships and model performance.

---

## Current Status
- Project proposal completed and submitted (November 3).
- Next steps: data cleaning, feature engineering, and model implementation.

---

## Data

This project uses the **S&P 500 ESG Risk Ratings** dataset from Kaggle.

- Dataset: https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings
- Source: Kaggle (originally based on Sustainalytics ESG data)

### How to obtain the data

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings
2. Extract the CSV file.
3. Place the file in the following directory



## Author
**Alessandra Failla**  
University of Lausanne, 2025