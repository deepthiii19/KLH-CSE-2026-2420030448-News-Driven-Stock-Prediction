# AI-Based Financial News Sentiment Analysis for Stock Market Prediction

An AI/ML project that combines financial news sentiment analysis with historical stock market data to predict stock price direction (UP or DOWN). The project uses **FinBERT** for financial sentiment analysis and a proposed **FinSent-Momentum Fusion (FSMF)** feature-engineering framework with a Gradient Boosting classifier.

## Overview

Financial markets are influenced by both quantitative market indicators and information published in financial news. This project explores how Natural Language Processing (NLP) and Machine Learning (ML) can combine these sources to classify stock market direction.

The system analyzes financial text, extracts sentiment scores, engineers sentiment and market-related features, and uses a machine learning model to predict upward or downward stock movement.

The project uses historical Apple Inc. (AAPL) data from the TRACE ACL18 dataset.

## Key Features

- **Financial Sentiment Analysis:** Uses FinBERT to analyze financial text and extract positive, negative, and neutral sentiment scores.
- **Daily Sentiment Aggregation:** Aggregates sentiment information from daily text files.
- **Feature Engineering:** Generates sentiment dynamics, market momentum, volatility, technical indicators, volume-related features, and sentiment–market interaction features.
- **FSMF Framework:** Combines sentiment and quantitative market information into an engineered feature representation.
- **Stock Direction Prediction:** Uses Gradient Boosting to classify market movement as UP or DOWN.
- **Model Evaluation:** Evaluates predictions using accuracy, precision, recall, and F1-score.
- **Interactive Interface:** Includes a Streamlit-based interface for demonstrating sentiment analysis and model predictions.

## Proposed Approach: FinSent-Momentum Fusion (FSMF)

FSMF is the proposed feature-engineering and feature-fusion framework developed for this project. It combines several categories of information:

| Feature Category | Description |
|---|---|
| Sentiment | Positive, negative, and neutral FinBERT scores |
| Sentiment Dynamics | Changes, momentum, and variability in sentiment |
| Market Momentum | Historical returns over multiple periods |
| Technical Indicators | RSI, MACD, and Bollinger Bands |
| Volatility | Rolling measures of market variability |
| Trading Volume | Volume changes, averages, and relative volume |
| Interaction Features | Relationships between sentiment and market indicators |

The engineered features are supplied to a Gradient Boosting classifier, which estimates the probability of upward stock movement.

**Note:** FSMF is a feature-fusion framework, not a newly invented standalone machine learning algorithm.

## System Workflow

1. Load historical AAPL market data, daily financial text, and stock movement labels.
2. Preprocess and align records by date.
3. Apply FinBERT to extract financial sentiment scores.
4. Aggregate sentiment scores at the daily level.
5. Merge sentiment, market data, and target labels.
6. Generate FSMF features, including sentiment dynamics and technical indicators.
7. Split the dataset chronologically into training, validation, and test sets.
8. Train the Gradient Boosting classifier.
9. Evaluate model performance on held-out data.
10. Present sentiment analysis and predictions through the Streamlit interface.

## Technologies Used

- **Programming Language:** Python
- **NLP Model:** FinBERT (`ProsusAI/finbert`)
- **Machine Learning:** Scikit-learn, Gradient Boosting
- **Data Processing:** Pandas, NumPy
- **Technical Analysis:** RSI, MACD, Bollinger Bands, and rolling market statistics
- **Model Persistence:** Joblib / Pickle
- **Web Application:** Streamlit
- **Dataset:** TRACE ACL18 AAPL dataset

## Preliminary Results

The current FSMF experiment produced the following results on its held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 69.12% |
| Precision | 61.90% |
| Recall | 83.87% |
| F1-score | 71.23% |

These results are preliminary and apply to the current experimental dataset split. They do not guarantee future stock market performance.

The initial Logistic Regression baseline achieved 56% test accuracy on a different dataset size and split. A controlled comparison using identical data and test samples is needed before concluding that FSMF provides a direct performance improvement.

## Dataset

The project uses the TRACE ACL18 dataset for AAPL, containing historical market information, daily text files, and stock movement labels.

The preprocessing pipeline generates intermediate and final datasets, including:

- `AAPL_model_dataset.csv` — merged market, text-derived sentiment, and label data.
- `AAPL_daily_sentiment.csv` — daily aggregated sentiment scores.
- `AAPL_features.csv` — baseline engineered features.
- `AAPL_FSMF_features.csv` — FSMF feature dataset.

**Dataset note:** The source market data contains normalized/scaled values. These values should not be interpreted as actual historical dollar prices without applying the appropriate transformation.

Please review the dataset's original license and usage terms before redistributing it.

## Project Structure

```text
NewsDrivenStockPrediction/
│
├── data/
│   ├── TRACE_ACL18_joint_prediction_model_set/
│   ├── AAPL_model_dataset.csv
│   ├── AAPL_daily_sentiment.csv
│   ├── AAPL_features.csv
│   └── AAPL_FSMF_features.csv
│
├── src/
│   ├── preprocessing.py
│   ├── build_sentiment_features.py
│   ├── build_model_dataset.py
│   ├── feature_engineering.py
│   ├── proposed_features.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── fsmf_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

*The structure above is representative. Adjust filenames to match the files actually present in your repository.*

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NewsDrivenStockPrediction
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python virtual-environment documentation or use the environment's Python executable directly.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the libraries used by your scripts, such as:

```text
pandas
numpy
scikit-learn
transformers
torch
streamlit
joblib
matplotlib
seaborn
```

Review and pin compatible package versions before publishing for reproducibility.

## Running the Project

Run the scripts from the project root in the order supported by your implementation.

### Step 1: Preprocess the dataset

```powershell
python src/preprocessing.py
```

### Step 2: Generate sentiment features

```powershell
python src/build_sentiment_features.py
```

This step loads FinBERT and may take considerable time, particularly on a CPU.

### Step 3: Build the merged model dataset

```powershell
python src/build_model_dataset.py
```

### Step 4: Generate baseline features

```powershell
python src/feature_engineering.py
```

### Step 5: Generate FSMF features

```powershell
python src/proposed_features.py
```

### Step 6: Train and evaluate the model

Run the training and evaluation scripts corresponding to your current implementation:

```powershell
python src/train_model.py
python src/evaluate_model.py
```

**Important:** Verify that these scripts train and evaluate the intended FSMF model before using their outputs as FSMF results. The filenames alone do not guarantee which dataset or model each script uses.

### Step 7: Launch the Streamlit application

If your application entry point is `app.py`, run:

```powershell
streamlit run app.py
```

If your application file has a different name or location, replace `app.py` with the correct path.

## Evaluation Metrics

The project uses the following evaluation metrics:

- **Accuracy:** Fraction of all predictions classified correctly.
- **Precision:** Fraction of predicted UP movements that are actually UP.
- **Recall:** Fraction of actual UP movements correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

Chronological data splitting is used to respect the time ordering of financial observations. Further experiments, such as feature ablation, feature importance analysis, and repeated time-series evaluation, can help establish the contribution and robustness of the proposed framework.



We acknowledge the researchers and dataset creators whose work supports financial sentiment analysis and stock movement prediction. FinBERT and the TRACE ACL18 dataset are important components of this project.

