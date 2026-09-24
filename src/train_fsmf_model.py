import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "AAPL_FSMF_features.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "fsmf_model.pkl"


def main():

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Features used by FSMF
    # ---------------------------------------------------------

    features = [
        "positive_score",
        "negative_score",
        "neutral_score",
        "tweet_count",

        "sentiment_score",
        "sentiment_pressure",
        "sentiment_change_1d",
        "sentiment_change_3d",
        "sentiment_momentum_5d",
        "sentiment_acceleration",
        "sentiment_std_5",
        "sentiment_std_10",

        "tweet_count_change",
        "tweet_count_mean_5",
        "tweet_count_std_5",

        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",

        "volatility_5d",
        "volatility_10d",

        "rsi_14",

        "macd",
        "macd_signal",
        "macd_histogram",

        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",

        "volume_change",
        "volume_mean_5",
        "volume_ratio",

        "sentiment_x_momentum",
        "sentiment_x_volatility",
        "sentiment_x_volume"
    ]

    target = "Target1d"

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    train = df.iloc[:train_end]

    validation = df.iloc[train_end:val_end]

    test = df.iloc[val_end:]

    print("\n===== FSMF DATA SPLIT =====")

    print(
        f"Training:   {len(train)} rows "
        f"({train['Date'].iloc[0].date()} → "
        f"{train['Date'].iloc[-1].date()})"
    )

    print(
        f"Validation: {len(validation)} rows "
        f"({validation['Date'].iloc[0].date()} → "
        f"{validation['Date'].iloc[-1].date()})"
    )

    print(
        f"Test:       {len(test)} rows "
        f"({test['Date'].iloc[0].date()} → "
        f"{test['Date'].iloc[-1].date()})"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )

    print("\nTraining FinSent-Momentum Fusion model...")

    model.fit(
        train[features],
        train[target]
    )

    # ---------------------------------------------------------
    # Evaluation function
    # ---------------------------------------------------------

    def evaluate(name, data):

        predictions = model.predict(
            data[features]
        )

        accuracy = accuracy_score(
            data[target],
            predictions
        )

        precision = precision_score(
            data[target],
            predictions,
            zero_division=0
        )

        recall = recall_score(
            data[target],
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            data[target],
            predictions,
            zero_division=0
        )

        print(f"\n===== {name.upper()} RESULTS =====")

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    evaluate("Validation", validation)

    evaluate("Test", test)

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "model_name": "FinSent-Momentum Fusion"
        },
        MODEL_FILE
    )

    print(
        f"\nFSMF model saved to:\n{MODEL_FILE}"
    )


if __name__ == "__main__":

    main()