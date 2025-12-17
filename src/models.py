from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test 
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train a logistic regression
    Return the model and the scaler 
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest Model (non linear)
    """
    rf = RandomForestClassifier(
        n_estimators=400, #more estimators => more stable model 
        random_state=42,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    return rf


