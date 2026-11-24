
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "split"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    # Load parameters
    params = load_params()

    test_size = params["split"]["test_size"]
    n_bins = params["split"]["n_bins"]
    seed = params["split"]["seed"]

    # Read processed dataset
    df = pd.read_csv(INPUT_PATH)

    print("Processed dataset shape:", df.shape)

    # Separate features and target
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Create bins for stratified splitting
    y_bins = pd.qcut(
        y,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y_bins
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the four datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    y_train.to_frame(name="SalePrice").to_csv(
        OUTPUT_DIR / "y_train.csv", index=False
    )
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_test.to_frame(name="SalePrice").to_csv(
        OUTPUT_DIR / "y_test.csv", index=False
    )

    print("\nDataset split completed successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("\nFiles saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
