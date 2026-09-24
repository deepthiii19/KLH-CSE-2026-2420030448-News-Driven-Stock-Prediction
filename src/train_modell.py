import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "AAPL_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "logistic_regression_model.pkl"


def main():

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    features = [
        "positive_score",
        "negative_score",
        "neutral_score",
        "tweet_count",
        "sentiment_score",
        "movement_lag1",
        "movement_lag2",
        "movement_lag3",
        "movement_mean_3",
        "movement_mean_7",
        "movement_std_7",
        "sentiment_lag1",
        "sentiment_mean_3",
        "volume_change",
    ]

    target = "Target1d"

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("\nTraining final Logistic Regression model...")

    model.fit(train[features], train[target])

    validation_accuracy = model.score(
        validation[features],
        validation[target]
    )

    test_accuracy = model.score(
        test[features],
        test[target]
    )

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features
        },
        MODEL_FILE
    )

    print("\n===== FINAL MODEL =====")
    print("Model: Logistic Regression")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()