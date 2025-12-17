from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


def ensure_results_dir(results_dir: str = "results") -> Path:
    """
    Create results/ if it doesn't exist. 
    Make sure that the code doesn't crash if results/ does not exist.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Save resultst in a JSON file.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def classification_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    """
    Measures performance of classification models. 
    weighted = if classes are not equilibrated 
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def run_classification_evaluation(
    model_name: str,
    y_true,
    y_pred,
    results_dir: str = "results",
    run_name: str = "run",
) -> Dict[str, Any]:
    """
    Calculate metrics and save it in a JSON file.
    """
    out_dir = ensure_results_dir(results_dir)
    prefix = f"{run_name}_{model_name}"

    metrics = classification_metrics(y_true, y_pred)
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "model": model_name,
        "task_type": "classification",
        **metrics,
    }

    save_json(payload, out_dir / f"{prefix}_metrics.json")


    return payload