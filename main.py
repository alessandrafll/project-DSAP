from pathlib import Path

from src.data_loader import build_dataset
from src.models import split_data, train_logistic_regression, train_random_forest
from src.evaluation import run_classification_evaluation



def main() -> None:
    # 1) Path to data/raw and construction of metrics
    data_dir = Path("data")
    X, y, _df_merged = build_dataset(data_dir, use_cache=True, refresh_cache=False)


    # 3) Split train / test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4) Model 1 : Logistic Regression with scaler 
    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = logreg_model.predict(X_test_scaled)

    lr_metrics = run_classification_evaluation(
        model_name="logistic_regression",
        y_true=y_test,
        y_pred=y_pred_lr,
        results_dir="results",
        run_name="baseline",
    )
    print("Logistic Regression metrics:", lr_metrics)

    # 5) Model 2 : Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_metrics = run_classification_evaluation(
        model_name="random_forest",
        y_true=y_test,
        y_pred=y_pred_rf,
        results_dir="results",
        run_name="baseline",
    )
    print("Random Forest metrics:", rf_metrics)


if __name__ == "__main__":
    main()