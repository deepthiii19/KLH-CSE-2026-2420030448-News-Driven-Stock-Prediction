import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "AAPL_model_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "AAPL_FSMF_features.csv"


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-10)

    return 100 - (100 / (1 + rs))


def create_features():

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------
    # Base sentiment
    # ---------------------------------------------------------

    df["sentiment_score"] = (
        df["positive_score"] - df["negative_score"]
    )

    # Normalized sentiment pressure
    df["sentiment_pressure"] = (
        (df["positive_score"] - df["negative_score"])
        /
        (
            df["positive_score"]
            + df["negative_score"]
            + 1e-6
        )
    )

    # ---------------------------------------------------------
    # Sentiment dynamics
    # ---------------------------------------------------------

    df["sentiment_change_1d"] = (
        df["sentiment_score"].diff()
    )

    df["sentiment_change_3d"] = (
        df["sentiment_score"]
        - df["sentiment_score"].shift(3)
    )

    df["sentiment_momentum_5d"] = (
        df["sentiment_score"]
        - df["sentiment_score"].shift(5)
    )

    df["sentiment_acceleration"] = (
        df["sentiment_change_1d"]
        - df["sentiment_change_1d"].shift(1)
    )

    df["sentiment_std_5"] = (
        df["sentiment_score"]
        .rolling(5)
        .std()
    )

    df["sentiment_std_10"] = (
        df["sentiment_score"]
        .rolling(10)
        .std()
    )

    # ---------------------------------------------------------
    # News intensity
    # ---------------------------------------------------------

    df["tweet_count_change"] = (
        df["tweet_count"].pct_change()
    )

    df["tweet_count_mean_5"] = (
        df["tweet_count"]
        .rolling(5)
        .mean()
    )

    df["tweet_count_std_5"] = (
        df["tweet_count"]
        .rolling(5)
        .std()
    )

    # ---------------------------------------------------------
    # Historical price returns
    # ---------------------------------------------------------

    df["return_1d"] = df["Close"].pct_change()

    df["return_3d"] = (
        df["Close"].pct_change(3)
    )

    df["return_5d"] = (
        df["Close"].pct_change(5)
    )

    df["return_10d"] = (
        df["Close"].pct_change(10)
    )

    # ---------------------------------------------------------
    # Price volatility
    # ---------------------------------------------------------

    df["volatility_5d"] = (
        df["return_1d"]
        .rolling(5)
        .std()
    )

    df["volatility_10d"] = (
        df["return_1d"]
        .rolling(10)
        .std()
    )

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        df["Close"],
        14
    )

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ---------------------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------------------

    bb_middle = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    df["bb_middle"] = bb_middle

    df["bb_upper"] = (
        bb_middle + 2 * bb_std
    )

    df["bb_lower"] = (
        bb_middle - 2 * bb_std
    )

    df["bb_bandwidth"] = (
        (df["bb_upper"] - df["bb_lower"])
        /
        (df["bb_middle"].abs() + 1e-6)
    )

    # ---------------------------------------------------------
    # Volume dynamics
    # ---------------------------------------------------------

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    df["volume_mean_5"] = (
        df["Volume"]
        .rolling(5)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        /
        (df["volume_mean_5"] + 1e-6)
    )

    # ---------------------------------------------------------
    # Sentiment-market interactions
    # ---------------------------------------------------------

    df["sentiment_x_momentum"] = (
        df["sentiment_score"]
        * df["return_5d"]
    )

    df["sentiment_x_volatility"] = (
        df["sentiment_score"]
        * df["volatility_5d"]
    )

    df["sentiment_x_volume"] = (
        df["sentiment_score"]
        * df["volume_ratio"]
    )

    # ---------------------------------------------------------
    # Remove missing values
    # ---------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna().reset_index(drop=True)

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


if __name__ == "__main__":

    df = create_features()

    print("\n===== FSMF FEATURE ENGINEERING =====")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nNew features:")

    for column in df.columns:
        print("-", column)

    print("\nTarget distribution:")
    print(df["Target1d"].value_counts())

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )