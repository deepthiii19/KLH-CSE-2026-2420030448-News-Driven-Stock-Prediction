import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

TRACE_DIR = (
    BASE_DIR
    / "data"
    / "TRACE_ACL18_joint_prediction_model_set"
)

PRICE_FILE = (
    TRACE_DIR
    / "data"
    / "prices"
    / "AAPL_price_panel.csv"
)

LABEL_FILE = (
    TRACE_DIR
    / "label"
    / "AAPL"
    / "labels.csv"
)

SENTIMENT_FILE = (
    BASE_DIR
    / "data"
    / "AAPL_daily_sentiment.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "AAPL_model_dataset.csv"
)


def load_data():

    prices = pd.read_csv(PRICE_FILE)
    labels = pd.read_csv(LABEL_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    prices["Date"] = pd.to_datetime(prices["Date"])
    labels["Date"] = pd.to_datetime(labels["Date"])
    sentiment["Date"] = pd.to_datetime(sentiment["Date"])

    return prices, labels, sentiment


def build_dataset():

    prices, labels, sentiment = load_data()

    # Merge price data with sentiment
    dataset = prices.merge(
        sentiment,
        on="Date",
        how="inner"
    )

    # Add TRACE target
    dataset = dataset.merge(
        labels,
        on="Date",
        how="inner"
    )

    # Sort chronologically
    dataset = (
        dataset
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return dataset


if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING FINAL AAPL MODEL DATASET")
    print("=" * 60)

    dataset = build_dataset()

    print("\nShape:")
    print(dataset.shape)

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nLast 5 rows:")
    print(dataset.tail())

    print("\nTarget distribution:")
    print(dataset["Target1d"].value_counts())

    print("\nMissing values:")
    print(dataset.isnull().sum())

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)