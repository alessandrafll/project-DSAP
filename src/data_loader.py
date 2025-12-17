from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# Raw Name of the csv ESG file in data/raw/
RAW_CSV_NAME = "esg_risk_ratings.csv"
PROCESSED_DIR_NAME = "processed"
YF_CACHE_NAME = "yf_cache.csv"


# Name of the target columns (what we want to predict)
TARGET_COL = "ESG_Risk_Level"

# Numerical columns that will be used as features 
NUM_COLS = ["MarketCap", "PE", "ROE", "DebtToEquity", "Beta", "DividendYield"]

# Categorical columns used for one-hot
CAT_COLS = ["Sector", "Industry"]


def load_esg_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Load the raw file ESG from data/raw/.
    """
    csv_path = raw_dir / RAW_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ESG CSV not found at {csv_path}. " #if the file doesn't exists 
            f"Please download it from Kaggle and place it in data/raw/" 
        )
    return pd.read_csv(csv_path) #reads the file with panda and return it as a DataFrame


def clean_esg(df_esg: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename the important ESG columns
    """
    df = df_esg.copy()

    # Rename 
    df = df.rename(columns={
        "Symbol": "Ticker",
        "Name": "Company",
        "Full Time Employees": "FullTimeEmployees",
        "Total ESG Risk score": "ESG_Risk_Score",
        "Environment Risk Score": "ESG_Env_Score",
        "Governance Risk Score": "ESG_Gov_Score",
        "Social Risk Score": "ESG_Soc_Score",
        "ESG Risk Level": "ESG_Risk_Level",
    })

    # Keep only useful data 
    cols_keep = [
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "FullTimeEmployees",
        "ESG_Risk_Score",
        "ESG_Env_Score",
        "ESG_Gov_Score",
        "ESG_Soc_Score",
        "ESG_Risk_Level",
        "ESG Risk Percentile",
        "Controversy Level",
        "Controversy Score",
        "Description",
    ]

    df = df[cols_keep].dropna(subset=["Ticker", "ESG_Risk_Score"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    return df


def _fetch_yf_info(ticker: str) -> dict:
    """
    Get some financial indicators for some tickers.
    If a error occurs, the function return "None"
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
    except Exception:
        # If yfinance crashes
        return {
            "Ticker": ticker,
            "MarketCap": None,
            "PE": None,
            "ROE": None,
            "DebtToEquity": None,
            "Beta": None,
            "DividendYield": None,
        }

    # If information is empty 
    if not isinstance(info, dict) or len(info) == 0:
        return {
            "Ticker": ticker,
            "MarketCap": None,
            "PE": None,
            "ROE": None,
            "DebtToEquity": None,
            "Beta": None,
            "DividendYield": None,
        }

    # Base case 
    return {
        "Ticker": ticker,
        "MarketCap": info.get("marketCap"),
        "PE": info.get("trailingPE"),
        "ROE": info.get("returnOnEquity"),
        "DebtToEquity": info.get("debtToEquity"),
        "Beta": info.get("beta"),
        "DividendYield": info.get("dividendYield"),
    }



def _processed_dir(base_data_dir: Path) -> Path:
    out = base_data_dir / PROCESSED_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_dataset(
    data_dir: Path,
    
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    - load and clean ESG data from data/raw/
    - get Yahoo Finance indicators 
    - merge
    - return X (features), y (target), df_merged
    """
    raw_dir = data_dir / "raw"

    # 1) ESG
    df_esg_raw = load_esg_raw(raw_dir)
    df_esg_clean = clean_esg(df_esg_raw)

    # 2) Merge ESG + finance
    df_merged = df_esg_clean.merge(df_yf, on="Ticker", how="inner")

    # 3) Imputation of missing values
    df_merged[NUM_COLS] = df_merged[NUM_COLS].apply(lambda col: col.fillna(col.median()))
    df_merged[CAT_COLS] = df_merged[CAT_COLS].fillna("Unknown")

    # 4) One-hot encoding
    X_num = df_merged[NUM_COLS]
    X_cat = pd.get_dummies(df_merged[CAT_COLS], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)

    # 5) Target
    y = df_merged[TARGET_COL]

    return X, y, df_merged