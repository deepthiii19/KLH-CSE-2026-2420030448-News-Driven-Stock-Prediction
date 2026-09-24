import pandas as pd
from pathlib import Path

# Find the project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 1. GOEMOTIONS DATASET
# ============================================================

print("=" * 60)
print("GOEMOTIONS DATASET")
print("=" * 60)

goemotions_path = BASE_DIR / "data" / "goemotions.csv"

goemotions = pd.read_csv(goemotions_path)

print("Shape:", goemotions.shape)

print("\nColumns:")
print(goemotions.columns.tolist())

print("\nFirst 5 rows:")
print(goemotions.head())

print("\nMissing values:")
print(goemotions.isnull().sum())


# ============================================================
# 2. TWEETEVAL SENTIMENT DATASET
# ============================================================

print("\n" + "=" * 60)
print("TWEETEVAL SENTIMENT DATASET")
print("=" * 60)

tweet_dir = BASE_DIR / "data" / "tweeteval_sentiment"

train_text = (tweet_dir / "train_text.txt").read_text(
    encoding="utf-8"
).splitlines()

train_labels = (tweet_dir / "train_labels.txt").read_text(
    encoding="utf-8"
).splitlines()

val_text = (tweet_dir / "val_text.txt").read_text(
    encoding="utf-8"
).splitlines()

val_labels = (tweet_dir / "val_labels.txt").read_text(
    encoding="utf-8"
).splitlines()

test_text = (tweet_dir / "test_text.txt").read_text(
    encoding="utf-8"
).splitlines()

test_labels = (tweet_dir / "test_labels.txt").read_text(
    encoding="utf-8"
).splitlines()

print("Training samples:", len(train_text))
print("Training labels:", len(train_labels))

print("Validation samples:", len(val_text))
print("Validation labels:", len(val_labels))

print("Test samples:", len(test_text))
print("Test labels:", len(test_labels))

print("\nFirst 5 training texts:")
for text in train_text[:5]:
    print("-", text)

print("\nFirst 5 training labels:")
print(train_labels[:5])

print("\nMapping:")
mapping_path = tweet_dir / "mapping.txt"
print(mapping_path.read_text(encoding="utf-8"))