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

    # Save confusion matrix figure
    save_confusion_matrix_png(
        y_true=y_true,
        y_pred=y_pred,
        out_path=out_dir / f"{prefix}_confusion_matrix.png",
        title=f"Confusion Matrix — {model_name}",
    )

    return payload

def save_metrics_summary(
    metrics_list: list[dict],
    results_dir: str = "results",
    run_name: str = "run",
) -> Path:
    """
    Transform a metrics dictionnary list into a table (CSV)
    """
    out_dir = ensure_results_dir(results_dir)
    df = pd.DataFrame(metrics_list)

    # order of columns
    preferred = ["run_name", "model", "task_type", "accuracy", "precision", "recall", "f1"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_path = out_dir / f"{run_name}_metrics_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path

def save_confusion_matrix_png(
    y_true,
    y_pred,
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a confusion matrix into a PNG iamge.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # write values into each cell
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_feature_importance_png(
    model,
    feature_names: list[str],
    out_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance (Random Forest)",
) -> None:
    """
    Save a PNG graph with importance of features from a Random Forest model.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model has no attribute 'feature_importances_'")

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # the most important feature on top
    ax.barh(df["feature"][::-1], df["importance"][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
