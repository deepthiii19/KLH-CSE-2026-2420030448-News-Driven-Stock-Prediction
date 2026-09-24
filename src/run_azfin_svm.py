"""
Schumaker & Chen (2009) AZFin-style baseline.

Adaptation to the TRACE AAPL dataset.

The original AZFin work used financial text together with
stock-related variables and an SVM-family classifier.

For this project:

    TRACE daily text
            +
    TRACE AAPL market variables
            ↓
         TF-IDF
            +
      price features
            ↓
          SVM
            ↓
        Target1d

IMPORTANT:
This is an adaptation to TRACE AAPL, NOT an exact reproduction
of the original AZFin dataset or experimental setup.

FSMF is not modified.
"""


from pathlib import Path
import warnings

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from scipy import sparse


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "AAPL_model_dataset.csv"
)

TWEETS_DIR = (
    ROOT
    / "data"
    / "TRACE_ACL18_joint_prediction_model_set"
    / "data"
    / "tweets"
    / "AAPL"
)

OUTPUT = (
    ROOT
    / "data"
    / "azfin_svm_results.csv"
)


# ============================================================
# SAME CHRONOLOGICAL PERIOD USED FOR FSMF
# ============================================================

TRAIN_END = pd.Timestamp(
    "2015-06-09"
)

VAL_END = pd.Timestamp(
    "2015-09-17"
)

TEST_START = pd.Timestamp(
    "2015-09-22"
)


# ============================================================
# LOAD DAILY TEXT
# ============================================================

def load_daily_text(date):

    filename = (
        TWEETS_DIR
        / f"{date.strftime('%Y-%m-%d')}.txt"
    )

    if not filename.exists():

        return ""

    try:

        return filename.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    dates
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {

        "model": model_name,

        "test_start": str(
            pd.Timestamp(
                dates.min()
            ).date()
        ),

        "test_end": str(
            pd.Timestamp(
                dates.max()
            ).date()
        ),

        "n_test": int(
            len(y_true)
        ),

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "tn": int(cm[0, 0]),

        "fp": int(cm[0, 1]),

        "fn": int(cm[1, 0]),

        "tp": int(cm[1, 1]),
    }


# ============================================================
# MAIN MODEL
# ============================================================

def run_azfin_svm(df):

    print()
    print("Loading daily TRACE AAPL text...")

    # --------------------------------------------------------
    # Load the raw tweet/news text corresponding to each date.
    # --------------------------------------------------------

    df = df.copy()

    df["text"] = df["Date"].apply(
        load_daily_text
    )

    # Count how many dates have text
    text_available = (
        df["text"].str.len() > 0
    )

    print(
        f"Dates with text: "
        f"{text_available.sum()} / {len(df)}"
    )

    if text_available.sum() == 0:

        raise RuntimeError(
            "\nNo TRACE text files were found.\n"
            f"Expected folder:\n{TWEETS_DIR}"
        )

    # --------------------------------------------------------
    # Keep rows that actually contain text.
    # --------------------------------------------------------

    df = df[
        text_available
    ].copy()

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_mask = (
        df["Date"] <= TRAIN_END
    )

    validation_mask = (
        (df["Date"] > TRAIN_END)
        &
        (df["Date"] <= VAL_END)
    )

    test_mask = (
        df["Date"] >= TEST_START
    )

    train = df.loc[
        train_mask
    ]

    validation = df.loc[
        validation_mask
    ]

    test = df.loc[
        test_mask
    ]

    print()
    print(
        f"Training rows:   {len(train)}"
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        f"Test rows:       {len(test)}"
    )

    # ========================================================
    # TEXT FEATURES
    # ========================================================

    print()
    print(
        "Building TF-IDF text features..."
    )

    vectorizer = TfidfVectorizer(

        lowercase=True,

        stop_words="english",

        ngram_range=(1, 2),

        min_df=2,

        max_features=10000,

        sublinear_tf=True,
    )

    # IMPORTANT:
    # Fit TF-IDF ONLY on training text.
    # This prevents test-period vocabulary information
    # from influencing training.

    X_text_train = vectorizer.fit_transform(
        train["text"]
    )

    X_text_test = vectorizer.transform(
        test["text"]
    )

    print(
        f"TF-IDF features: "
        f"{X_text_train.shape[1]}"
    )

    # ========================================================
    # PRICE / MARKET FEATURES
    # ========================================================

    print()
    print(
        "Preparing price features..."
    )

    price_features = [

        "MovementPercent",

        "Open",

        "High",

        "Low",

        "Close",

        "Volume",
    ]

    X_price_train = (
        train[price_features]
    )

    X_price_test = (
        test[price_features]
    )

    # --------------------------------------------------------
    # Scale numerical features using training data only.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_price_train_scaled = (
        scaler.fit_transform(
            X_price_train
        )
    )

    X_price_test_scaled = (
        scaler.transform(
            X_price_test
        )
    )

    # ========================================================
    # COMBINE TEXT + PRICE
    # ========================================================

    print()
    print(
        "Combining text and price features..."
    )

    X_train = sparse.hstack(

        [
            X_text_train,

            sparse.csr_matrix(
                X_price_train_scaled
            ),
        ],

        format="csr"
    )

    X_test = sparse.hstack(

        [
            X_text_test,

            sparse.csr_matrix(
                X_price_test_scaled
            ),
        ],

        format="csr"
    )

    # ========================================================
    # TARGET
    # ========================================================

    y_train = (
        train["Target1d"]
        .astype(int)
    )

    y_test = (
        test["Target1d"]
        .astype(int)
    )

    # ========================================================
    # TRAIN SVM
    # ========================================================

    print()
    print(
        "Training AZFin-style SVM..."
    )

    model = SVC(

        kernel="linear",

        C=1.0,

        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICT
    # ========================================================

    print()
    print(
        "Predicting test set..."
    )

    predictions = model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    result = calculate_metrics(

        "Schumaker & Chen (2009) - AZFin-style Text+Price SVM",

        y_test.to_numpy(),

        predictions,

        test["Date"].to_numpy()
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)

    print(
        "SCHUMAKER & CHEN (2009) - AZFIN-STYLE SVM"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA}"
        )

    # --------------------------------------------------------
    # Check raw text folder
    # --------------------------------------------------------

    if not TWEETS_DIR.exists():

        raise FileNotFoundError(

            "\nTRACE AAPL tweet folder not found:\n"

            f"{TWEETS_DIR}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()

    print(
        f"Loading dataset:\n{DATA}"
    )

    df = pd.read_csv(
        DATA
    )

    # Convert dates
    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # Sort chronologically
    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Required columns
    #
    # NOTICE:
    # We DO NOT require a text column in AAPL_model_dataset.csv.
    #
    # We reconstruct text from the original TRACE files.
    # --------------------------------------------------------

    required = {

        "Date",

        "MovementPercent",

        "Open",

        "High",

        "Low",

        "Close",

        "Volume",

        "Target1d",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    result = run_azfin_svm(
        df
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_df = pd.DataFrame(
        [result]
    )

    output_df.to_csv(
        OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 65)

    print("RESULT")

    print("=" * 65)

    print()

    print(
        output_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()

    print("=" * 65)

    print(
        f"Results saved to:\n{OUTPUT}"
    )

    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()