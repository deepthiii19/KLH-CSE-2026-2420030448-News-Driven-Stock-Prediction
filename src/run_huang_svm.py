"""
Huang et al. (2005)-style Technical SVM Baseline

This is an adaptation of the Huang et al. stock-movement SVM approach
to the TRACE AAPL dataset used in this project.

This is NOT an exact reproduction of the original paper's dataset
or experimental setup.

FSMF is not modified.
"""

from pathlib import Path
import warnings

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "AAPL_model_dataset.csv"

OUTPUT = ROOT / "data" / "huang_svm_results.csv"


# ============================================================
# FSMF-ALIGNED CHRONOLOGICAL SPLIT
# ============================================================

TRAIN_END = pd.Timestamp("2015-06-09")

VAL_END = pd.Timestamp("2015-09-17")

TEST_START = pd.Timestamp("2015-09-22")


# ============================================================
# CREATE TECHNICAL FEATURES
# ============================================================

def add_technical_features(df):

    d = df.copy()

    # Previous-day movement
    d["movement_lag1"] = (
        d["MovementPercent"].shift(1)
    )

    # Two-days-ago movement
    d["movement_lag2"] = (
        d["MovementPercent"].shift(2)
    )

    # Three-days-ago movement
    d["movement_lag3"] = (
        d["MovementPercent"].shift(3)
    )

    # 3-day average movement
    d["movement_mean_3"] = (
        d["MovementPercent"]
        .rolling(3)
        .mean()
    )

    # 7-day average movement
    d["movement_mean_7"] = (
        d["MovementPercent"]
        .rolling(7)
        .mean()
    )

    # 7-day movement volatility
    d["movement_std_7"] = (
        d["MovementPercent"]
        .rolling(7)
        .std()
    )

    # Volume change
    d["volume_change"] = (
        d["Volume"]
        .pct_change()
    )

    # 3-day cumulative movement
    d["return_3d"] = (
        d["MovementPercent"]
        .rolling(3)
        .sum()
    )

    # 5-day volatility
    d["volatility_5d"] = (
        d["MovementPercent"]
        .rolling(5)
        .std()
    )

    return d


# ============================================================
# CALCULATE EVALUATION METRICS
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

    result = {

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

    return result


# ============================================================
# HUANG SVM
# ============================================================

def run_huang_svm(df):

    print()
    print("Preparing technical features...")

    # These are technical/market features only.
    #
    # IMPORTANT:
    # No text.
    # No FinBERT.
    # No sentiment scores.
    #
    # This is intentionally a traditional technical SVM baseline.

    features = [

        "movement_lag1",

        "movement_lag2",

        "movement_lag3",

        "movement_mean_3",

        "movement_mean_7",

        "movement_std_7",

        "volume_change",

        "return_3d",

        "volatility_5d",

    ]

    # Add technical features
    d = add_technical_features(df)

    # Remove rows where rolling features are unavailable
    d = d.dropna(
        subset=features + ["Target1d"]
    ).copy()

    print(
        f"Usable rows: {len(d)}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    train_mask = (
        d["Date"] <= TRAIN_END
    )

    validation_mask = (
        (d["Date"] > TRAIN_END)
        &
        (d["Date"] <= VAL_END)
    )

    test_mask = (
        d["Date"] >= TEST_START
    )

    train = d.loc[train_mask]

    validation = d.loc[validation_mask]

    test = d.loc[test_mask]

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
    # TRAINING DATA
    # ========================================================

    X_train = train[features]

    y_train = train["Target1d"].astype(int)

    # ========================================================
    # TEST DATA
    # ========================================================

    X_test = test[features]

    y_test = test["Target1d"].astype(int)

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print()
    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # TRAIN SVM
    # ========================================================

    print()
    print("Training Huang-style SVM...")

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # PREDICT
    # ========================================================

    print(
        "Predicting test set..."
    )

    predictions = model.predict(
        X_test_scaled
    )

    # ========================================================
    # METRICS
    # ========================================================

    result = calculate_metrics(

        "Huang et al. (2005) - Technical SVM",

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
    print("=" * 60)
    print(
        "HUANG ET AL. (2005) - TECHNICAL SVM BASELINE"
    )
    print("=" * 60)

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not DATA.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA}"
        )

    print()
    print(
        f"Loading dataset:\n{DATA}"
    )

    # Load dataset
    df = pd.read_csv(DATA)

    # Convert date
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

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    # IMPORTANT:
    # There is NO "text" here.
    # There are NO sentiment columns here.

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
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # RUN MODEL
    # ========================================================

    result = run_huang_svm(df)

    # Convert result into DataFrame
    output_df = pd.DataFrame(
        [result]
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_df.to_csv(
        OUTPUT,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print()

    print(
        output_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 60)

    print(
        f"Results saved to:\n{OUTPUT}"
    )

    print("=" * 60)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()