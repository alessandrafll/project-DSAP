from pathlib import Path

from src.data_loader import build_dataset


def main() -> None:
    # 1) Path to data/raw and construction of metrics
    data_dir = Path("data")
    X, y, _df_merged = build_dataset(data_dir, use_cache=True, refresh_cache=False)