
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "split"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def main():
    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["SalePrice"]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create models directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("\nModel trained successfully!")
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
