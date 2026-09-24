import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

TWEET_DIR = (
    BASE_DIR
    / "data"
    / "TRACE_ACL18_joint_prediction_model_set"
    / "data"
    / "tweets"
    / "AAPL"
)

OUTPUT_FILE = BASE_DIR / "data" / "AAPL_daily_sentiment.csv"

MODEL_NAME = "ProsusAI/finbert"


def create_sentiment_pipeline():
    print("Loading FinBERT...")

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None
    )


def analyze_daily_tweets(model, text):
    """
    Run FinBERT on all tweets for one day
    and return average sentiment probabilities.
    """

    tweets = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not tweets:
        return {
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 0.0,
            "tweet_count": 0
        }

    results = model(
        tweets,
        batch_size=8,
        truncation=True
    )

    positive_scores = []
    negative_scores = []
    neutral_scores = []

    for result in results:

        scores = {
            item["label"].lower(): item["score"]
            for item in result
        }

        positive_scores.append(
            scores.get("positive", 0.0)
        )

        negative_scores.append(
            scores.get("negative", 0.0)
        )

        neutral_scores.append(
            scores.get("neutral", 0.0)
        )

    return {
        "positive_score": sum(positive_scores) / len(positive_scores),
        "negative_score": sum(negative_scores) / len(negative_scores),
        "neutral_score": sum(neutral_scores) / len(neutral_scores),
        "tweet_count": len(tweets)
    }


def build_daily_sentiment():

    model = create_sentiment_pipeline()

    tweet_files = sorted(TWEET_DIR.glob("*.txt"))

    if not tweet_files:
        raise ValueError(
            f"No tweet files found in {TWEET_DIR}"
        )

    rows = []

    print(f"\nFound {len(tweet_files)} daily tweet files.")
    print("Processing AAPL tweets with FinBERT...\n")

    for tweet_file in tqdm(tweet_files):

        date = pd.to_datetime(tweet_file.stem)

        text = tweet_file.read_text(
            encoding="utf-8",
            errors="replace"
        )

        sentiment = analyze_daily_tweets(
            model,
            text
        )

        rows.append({
            "Date": date,
            **sentiment
        })

    df = pd.DataFrame(rows)

    df = df.sort_values("Date").reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return df


if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING DAILY FINBERT SENTIMENT FEATURES")
    print("=" * 60)

    df = build_daily_sentiment()

    print("\n" + "=" * 60)
    print("SENTIMENT FEATURES CREATED")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nSaved to:")
    print(OUTPUT_FILE)