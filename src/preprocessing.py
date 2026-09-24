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

TWEET_DIR = (
    TRACE_DIR
    / "data"
    / "tweets"
    / "AAPL"
)

LABEL_FILE = (
    TRACE_DIR
    / "label"
    / "AAPL"
    / "labels.csv"
)


def load_price_data():
    """
    Load AAPL daily price data from TRACE.
    """

    df = pd.read_csv(PRICE_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    # TRACE files are stored newest -> oldest.
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_label_data():
    """
    Load AAPL 1-day movement labels from TRACE.
    """

    df = pd.read_csv(LABEL_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_daily_tweets():
    """
    Load all AAPL daily tweet files.

    Each file is named:
        YYYY-MM-DD.txt

    Returns one row per date.
    """

    rows = []

    for tweet_file in sorted(TWEET_DIR.glob("*.txt")):

        date_string = tweet_file.stem

        try:
            date = pd.to_datetime(date_string)
        except ValueError:
            print(f"Skipping invalid tweet filename: {tweet_file.name}")
            continue

        text = tweet_file.read_text(
            encoding="utf-8",
            errors="replace"
        )

        rows.append({
            "Date": date,
            "text": text
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No AAPL tweet files were found.")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def create_dataset():
    """
    Merge TRACE prices, daily tweets, and labels by date.
    """

    prices = load_price_data()
    tweets = load_daily_tweets()
    labels = load_label_data()

    dataset = prices.merge(
        tweets,
        on="Date",
        how="inner"
    )

    dataset = dataset.merge(
        labels,
        on="Date",
        how="inner"
    )

    dataset = dataset.sort_values("Date").reset_index(drop=True)

    return dataset


if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING TRACE AAPL DATASET")
    print("=" * 60)

    prices = load_price_data()
    tweets = load_daily_tweets()
    labels = load_label_data()

    print("\nPrices:")
    print(prices.shape)
    print(prices["Date"].min(), "->", prices["Date"].max())

    print("\nTweets:")
    print(tweets.shape)
    print(tweets["Date"].min(), "->", tweets["Date"].max())

    print("\nLabels:")
    print(labels.shape)
    print(labels["Date"].min(), "->", labels["Date"].max())

    dataset = create_dataset()

    print("\n" + "=" * 60)
    print("MERGED DATASET")
    print("=" * 60)

    print("Shape:", dataset.shape)

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nFirst 3 rows:")
    print(
        dataset[
            [
                "Date",
                "MovementPercent",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Target1d"
            ]
        ].head(3)
    )

    print("\nTarget distribution:")
    print(dataset["Target1d"].value_counts())

    print("\nMissing values:")
    print(dataset.isnull().sum())