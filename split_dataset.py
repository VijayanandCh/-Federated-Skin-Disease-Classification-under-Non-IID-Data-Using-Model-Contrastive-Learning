import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\Major Project\HAM10000"

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

METADATA_PATH = os.path.join(
    RAW_DIR,
    "HAM10000_metadata.csv"
)

# Create processed folder if it does not exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING LESION-AWARE DATASET SPLIT")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)

print("\nTotal images:", len(df))
print("Total lesions:", df["lesion_id"].nunique())

# ============================================================
# CREATE ONE ROW PER LESION
# ============================================================

# Each lesion gets one class label.
# This allows us to split lesions instead of individual images.

lesion_df = (
    df.groupby("lesion_id", as_index=False)
    .agg(dx=("dx", "first"))
)

print("\nUnique lesions:", len(lesion_df))

# ============================================================
# FIRST SPLIT: 70% TRAIN, 30% TEMPORARY
# ============================================================

train_lesions, temp_lesions = train_test_split(
    lesion_df,
    test_size=0.30,
    stratify=lesion_df["dx"],
    random_state=42
)

# ============================================================
# SECOND SPLIT: 15% VALIDATION, 15% TEST
# ============================================================

validation_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=0.50,
    stratify=temp_lesions["dx"],
    random_state=42
)

# ============================================================
# GET IMAGE RECORDS FOR EACH SPLIT
# ============================================================

train_df = df[
    df["lesion_id"].isin(train_lesions["lesion_id"])
].copy()

validation_df = df[
    df["lesion_id"].isin(validation_lesions["lesion_id"])
].copy()

test_df = df[
    df["lesion_id"].isin(test_lesions["lesion_id"])
].copy()

# ============================================================
# SAVE CSV FILES
# ============================================================

train_path = os.path.join(
    PROCESSED_DIR,
    "train.csv"
)

validation_path = os.path.join(
    PROCESSED_DIR,
    "validation.csv"
)

test_path = os.path.join(
    PROCESSED_DIR,
    "test.csv"
)

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nSPLIT RESULTS")
print("-" * 60)

print(
    "Training:",
    len(train_df),
    "images |",
    train_df["lesion_id"].nunique(),
    "lesions"
)

print(
    "Validation:",
    len(validation_df),
    "images |",
    validation_df["lesion_id"].nunique(),
    "lesions"
)

print(
    "Test:",
    len(test_df),
    "images |",
    test_df["lesion_id"].nunique(),
    "lesions"
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTRAINING CLASS DISTRIBUTION")
print(train_df["dx"].value_counts())

print("\nVALIDATION CLASS DISTRIBUTION")
print(validation_df["dx"].value_counts())

print("\nTEST CLASS DISTRIBUTION")
print(test_df["dx"].value_counts())

# ============================================================
# CHECK FOR LESION LEAKAGE
# ============================================================

train_ids = set(train_df["lesion_id"])
validation_ids = set(validation_df["lesion_id"])
test_ids = set(test_df["lesion_id"])

print("\nLESION LEAKAGE CHECK")
print("-" * 60)

print(
    "Train ∩ Validation:",
    len(train_ids & validation_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(validation_ids & test_ids)
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)

print("\nFiles created:")
print(train_path)
print(validation_path)
print(test_path)