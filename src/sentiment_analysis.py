from transformers import pipeline


MODEL_NAME = "ProsusAI/finbert"


def create_sentiment_pipeline():
    """
    Load the FinBERT sentiment analysis model.
    """

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None
    )


def analyze_texts(sentiment_model, texts):
    """
    Analyze a list of texts using FinBERT.

    Returns sentiment probabilities for:
    positive, negative, and neutral.
    """

    results = sentiment_model(texts)

    processed_results = []

    for text, result in zip(texts, results):

        scores = {
            item["label"].lower(): item["score"]
            for item in result
        }

        processed_results.append({
            "text": text,
            "sentiment": max(
                scores,
                key=scores.get
            ),
            "positive_score": scores.get("positive", 0.0),
            "negative_score": scores.get("negative", 0.0),
            "neutral_score": scores.get("neutral", 0.0)
        })

    return processed_results


if __name__ == "__main__":

    print("=" * 60)
    print("TESTING FINBERT SENTIMENT FEATURES")
    print("=" * 60)

    model = create_sentiment_pipeline()

    test_texts = [
        "The company reported strong quarterly earnings.",
        "The company announced a major loss this quarter.",
        "The company released its quarterly financial report."
    ]

    results = analyze_texts(model, test_texts)

    for result in results:

        print("\nText:", result["text"])
        print("Sentiment:", result["sentiment"])
        print(
            "Positive:",
            round(result["positive_score"], 4)
        )
        print(
            "Negative:",
            round(result["negative_score"], 4)
        )
        print(
            "Neutral:",
            round(result["neutral_score"], 4)
        )