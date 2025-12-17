from pathlib import Path

from src.data_loader import build_dataset
from src.models import split_data, train_logistic_regression, train_random_forest


def main():
    data_dir = Path("data")
    X, y, _ = build_dataset(data_dir)

    X_train, X_test, y_train, y_test = split_data(X, y)

    rf = train_random_forest(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Random Forest OK")
    print("Predictions shape:", y_pred.shape)
    print("Unique predictions:", set(y_pred))


if __name__ == "__main__":
    main()
