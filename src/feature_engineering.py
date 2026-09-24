import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "AAPL_model_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "AAPL_features.csv"


def create_features():

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    # Make sure data is chronological
    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Sentiment features
    # ---------------------------------------------------------

    df["sentiment_score"] = (
        df["positive_score"] - df["negative_score"]
    )

    # ---------------------------------------------------------
    # Historical market features
    # ---------------------------------------------------------

    df["movement_lag1"] = df["MovementPercent"].shift(1)

    df["movement_lag2"] = df["MovementPercent"].shift(2)

    df["movement_lag3"] = df["MovementPercent"].shift(3)

    # Rolling statistics using only previous days
    df["movement_mean_3"] = (
        df["MovementPercent"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["movement_mean_7"] = (
        df["MovementPercent"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["movement_std_7"] = (
        df["MovementPercent"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    # ---------------------------------------------------------
    # Historical sentiment features
    # ---------------------------------------------------------

    df["sentiment_lag1"] = df["sentiment_score"].shift(1)

    df["sentiment_mean_3"] = (
        df["sentiment_score"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    # ---------------------------------------------------------
    # Volume feature
    # ---------------------------------------------------------

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    # ---------------------------------------------------------
    # Remove rows created by lag/rolling calculations
    # ---------------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    return df


if __name__ == "__main__":

    df = create_features()

    print("\nFeature engineering completed.")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nTarget distribution:")
    print(df["Target1d"].value_counts())

    print(f"\nSaved to: {OUTPUT_FILE}")