import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "AAPL_features.csv"


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

    model.fit(train[features], train[target])

    # Validation predictions
    val_predictions = model.predict(validation[features])

    # Test predictions
    test_predictions = model.predict(test[features])

    print("\n===== VALIDATION RESULTS =====")
    print(
        f"Accuracy:  "
        f"{accuracy_score(validation[target], val_predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(validation[target], val_predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(validation[target], val_predictions, zero_division=0):.4f}"
    )

    print(
        f"F1-score:  "
        f"{f1_score(validation[target], val_predictions, zero_division=0):.4f}"
    )

    print("\n===== TEST RESULTS =====")

    print(
        f"Accuracy:  "
        f"{accuracy_score(test[target], test_predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(test[target], test_predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(test[target], test_predictions, zero_division=0):.4f}"
    )

    print(
        f"F1-score:  "
        f"{f1_score(test[target], test_predictions, zero_division=0):.4f}"
    )

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(test[target], test_predictions))

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            test[target],
            test_predictions,
            target_names=["Down (0)", "Up (1)"],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()