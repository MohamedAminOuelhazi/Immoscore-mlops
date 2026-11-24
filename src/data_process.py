
from pathlib import Path

import pandas as pd
import yaml


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "data.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "dataset.csv"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():

    # Load parameters
    params = load_params()
    corr_threshold = params["process"]["corr_threshold"]

    # Load raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Keep numerical columns
    df = df.select_dtypes(include=["number"])

    # Remove rows with missing values
    df = df.dropna()

    print("After cleaning:", df.shape)

    # Calculate correlations with the target
    correlations = df.corr()["SalePrice"].drop("SalePrice")

    # Select features based on absolute correlation
    selected_features = correlations[
        correlations.abs() > corr_threshold
    ].index.tolist()

    # Keep selected features and target
    df = df[selected_features + ["SalePrice"]]

    print("Selected features:", len(selected_features))
    print("Processed dataset shape:", df.shape)

    # Save processed dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Processed dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
