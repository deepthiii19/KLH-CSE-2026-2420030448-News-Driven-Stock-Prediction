import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from transformers import pipeline


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "fsmf_model.pkl"
DATA_FILE = BASE_DIR / "data" / "AAPL_FSMF_features.csv"


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="FinSent-Momentum Fusion",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# Load FSMF model
# =========================================================

@st.cache_resource
def load_model():

    saved = joblib.load(MODEL_FILE)

    return (
        saved["model"],
        saved["features"],
        saved["model_name"]
    )


# =========================================================
# Load FinBERT
# =========================================================

@st.cache_resource
def load_finbert():

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None
    )


# =========================================================
# Load FSMF feature data
# =========================================================

@st.cache_data
def load_features():

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    return df.sort_values("Date").reset_index(drop=True)


# =========================================================
# FinBERT sentiment analysis
# =========================================================

def analyze_sentiment(model, text):

    result = model(text)[0]

    scores = {
        item["label"].lower(): item["score"]
        for item in result
    }

    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    neutral = scores.get("neutral", 0.0)

    sentiment = max(
        scores,
        key=scores.get
    )

    return sentiment, positive, negative, neutral


# =========================================================
# Application header
# =========================================================

st.title("📈 AI-Based Financial News Sentiment Analysis")

st.subheader(
    "FinSent-Momentum Fusion Model for Stock Market Prediction"
)

st.write(
    "This system analyzes financial news using FinBERT "
    "and combines sentiment information with historical "
    "market, momentum, volatility, technical-indicator, "
    "and volume features."
)


# =========================================================
# News input
# =========================================================

news_text = st.text_area(
    "Enter Financial News",
    placeholder=(
        "Example: Apple reports strong quarterly earnings "
        "and record revenue."
    ),
    height=120
)


# =========================================================
# Analyze button
# =========================================================

if st.button("Analyze News"):

    if not news_text.strip():

        st.warning(
            "Please enter a financial news statement."
        )

    else:

        # -------------------------------------------------
        # FinBERT
        # -------------------------------------------------

        with st.spinner(
            "Analyzing financial news with FinBERT..."
        ):

            finbert = load_finbert()

            (
                sentiment,
                positive,
                negative,
                neutral
            ) = analyze_sentiment(
                finbert,
                news_text
            )

        # -------------------------------------------------
        # Display sentiment
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("🧠 FinBERT Sentiment Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Sentiment",
            sentiment.upper()
        )

        col2.metric(
            "Positive",
            f"{positive:.2%}"
        )

        col3.metric(
            "Negative",
            f"{negative:.2%}"
        )

        col4.metric(
            "Neutral",
            f"{neutral:.2%}"
        )

        # -------------------------------------------------
        # Load FSMF model and features
        # -------------------------------------------------

        model, features, model_name = load_model()

        df = load_features()

        # Use latest available historical feature context
        latest = df.iloc[-1].copy()

        # -------------------------------------------------
        # Update sentiment features
        # -------------------------------------------------

        sentiment_score = positive - negative

        latest["positive_score"] = positive
        latest["negative_score"] = negative
        latest["neutral_score"] = neutral

        latest["sentiment_score"] = sentiment_score

        latest["sentiment_pressure"] = (
            (positive - negative)
            /
            (positive + negative + 1e-6)
        )

        # For the interactive demo, the new news is treated
        # as the current sentiment observation.
        latest["sentiment_change_1d"] = (
            sentiment_score
            - df["sentiment_score"].iloc[-2]
        )

        latest["sentiment_change_3d"] = (
            sentiment_score
            - df["sentiment_score"].iloc[-3]
        )

        latest["sentiment_momentum_5d"] = (
            sentiment_score
            - df["sentiment_score"].iloc[-5]
        )

        latest["sentiment_acceleration"] = (
            latest["sentiment_change_1d"]
            - df["sentiment_change_1d"].iloc[-1]
        )

        # -------------------------------------------------
        # Update news intensity
        # -------------------------------------------------

        # One entered news item
        latest["tweet_count"] = 1

        latest["tweet_count_change"] = (
            1 / (df["tweet_count"].iloc[-1] + 1e-6)
        ) - 1

        latest["tweet_count_mean_5"] = (
            df["tweet_count"]
            .tail(4)
            .tolist()
            + [1]
        )

        latest["tweet_count_mean_5"] = sum(
            latest["tweet_count_mean_5"]
        ) / 5

        # -------------------------------------------------
        # Recalculate interaction features
        # -------------------------------------------------

        latest["sentiment_x_momentum"] = (
            latest["sentiment_score"]
            * latest["return_5d"]
        )

        latest["sentiment_x_volatility"] = (
            latest["sentiment_score"]
            * latest["volatility_5d"]
        )

        latest["sentiment_x_volume"] = (
            latest["sentiment_score"]
            * latest["volume_ratio"]
        )

        # -------------------------------------------------
        # Prepare model input
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [
                [
                    latest[feature]
                    for feature in features
                ]
            ],
            columns=features
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        down_probability = probabilities[0]
        up_probability = probabilities[1]

        # -------------------------------------------------
        # Display prediction
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 FinSent-Momentum Fusion Prediction"
        )

        if prediction == 1:

            st.success(
                "📈 Predicted Market Direction: UP"
            )

        else:

            st.error(
                "📉 Predicted Market Direction: DOWN"
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Down Probability",
            f"{down_probability:.2%}"
        )

        col2.metric(
            "Up Probability",
            f"{up_probability:.2%}"
        )

        # -------------------------------------------------
        # Model information
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("⚙️ Model Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model",
            model_name
        )

        col2.metric(
            "Features",
            len(features)
        )

        col3.metric(
            "Test Accuracy",
            "69.12%"
        )

        st.caption(
            "The test accuracy is measured on the held-out "
            "TRACE AAPL test period. The displayed prediction "
            "probabilities are specific to the current input "
            "and are not the overall model accuracy."
        )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("Project Information")

    st.write("**Team Number:** 11")
    st.write("**Section:** 7")
    st.write("**Cluster:** 3")

    st.markdown("---")

    st.write(
        "**Sentiment Model:** FinBERT"
    )

    st.write(
        "**Proposed Model:** "
        "FinSent-Momentum Fusion"
    )

    st.write(
        "**Classifier:** Gradient Boosting"
    )

    st.markdown("---")

    st.write(
        "The proposed model combines financial sentiment "
        "dynamics, market momentum, technical indicators, "
        "volatility, volume behavior, and sentiment-market "
        "interaction features."
    )