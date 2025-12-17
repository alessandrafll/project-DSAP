from pathlib import Path
from src.data_loader import build_dataset

def main():
    data_dir = Path("data")

    X, y, df = build_dataset(data_dir, use_cache=True, refresh_cache=False)

    print("✅ build_dataset() OK")
    print("Merged df shape:", df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target classes:", sorted(y.dropna().unique()))
    print("First 5 columns of X:", list(X.columns[:5]))
    print("Cache exists:", (data_dir / "processed" / "yf_cache.csv").exists())

if __name__ == "__main__":
    main()