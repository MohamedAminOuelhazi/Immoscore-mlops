
from pathlib import Path
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "split"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
REPORT_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"


def main():
    # Load training and testing datasets
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["SalePrice"]

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["SalePrice"]

    # Load trained model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "train": {
            "RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "R2": float(r2_score(y_train, y_train_pred))
        },
        "test": {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "R2": float(r2_score(y_test, y_test_pred))
        }
    }

    # Display results
    print("Model Evaluation Results")
    print("------------------------")
    print(f"Train RMSE: {metrics['train']['RMSE']:.2f}")
    print(f"Train R2:   {metrics['train']['R2']:.4f}")
    print(f"Test RMSE:  {metrics['test']['RMSE']:.2f}")
    print(f"Test R2:    {metrics['test']['R2']:.4f}")

    # Save metrics
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nMetrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    main()
