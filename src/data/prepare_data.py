
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "m1")

METADATA_PATH = os.path.join(
    RAW_DIR, "HAM10000_metadata.csv"
)

IMAGE_DIRS = [
    os.path.join(RAW_DIR, "HAM10000_images_part_1"),
    os.path.join(RAW_DIR, "HAM10000_images_part_2")
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD METADATA
# --------------------------------------------------

print("\n========== LOADING HAM10000 ==========")

df = pd.read_csv(METADATA_PATH)

required_columns = [
    "image_id",
    "lesion_id",
    "dx"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required metadata column missing: {column}"
        )

print("Dataset shape:", df.shape)
print("Metadata columns:", list(df.columns))


# --------------------------------------------------
# 3. BASIC DATA VALIDATION
# --------------------------------------------------

print("\n========== DATA VALIDATION ==========")

print("\nMissing values:")
print(df[required_columns].isnull().sum())

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in image_id, lesion_id or dx. "
        "Inspect the metadata before proceeding."
    )

df["image_id"] = df["image_id"].astype(str)
df["lesion_id"] = df["lesion_id"].astype(str)
df["dx"] = df["dx"].astype(str).str.strip().str.lower()

duplicate_count = df["image_id"].duplicated().sum()

print("\nDuplicate image IDs:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate image IDs found. Resolve them before splitting."
    )

print("Unique lesions:", df["lesion_id"].nunique())

# Every lesion should have one diagnosis label.
lesion_label_counts = df.groupby("lesion_id")["dx"].nunique()

inconsistent_lesions = lesion_label_counts[
    lesion_label_counts > 1
]

if len(inconsistent_lesions) > 0:
    raise ValueError(
        "Some lesion IDs have multiple diagnosis labels. "
        "Inspect these records before splitting."
    )


# --------------------------------------------------
# 4. VERIFY IMAGE FILES
# --------------------------------------------------

print("\n========== VERIFYING IMAGE FILES ==========")

image_lookup = {}

for folder in IMAGE_DIRS:

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_id = os.path.splitext(filename)[0]

            if image_id in image_lookup:
                raise ValueError(
                    f"Duplicate image filename found: {image_id}"
                )

            image_lookup[image_id] = os.path.join(
                folder, filename
            )

metadata_ids = set(df["image_id"])
available_ids = set(image_lookup.keys())

missing_images = metadata_ids - available_ids
extra_images = available_ids - metadata_ids

print("Metadata images:", len(metadata_ids))
print("Images found:", len(available_ids))
print("Missing images:", len(missing_images))
print("Extra images:", len(extra_images))

if missing_images:
    print("Missing image examples:", list(missing_images)[:10])
    raise ValueError(
        "Some metadata records do not have corresponding images."
    )

if extra_images:
    print(
        "Note: Extra images not referenced in metadata:",
        len(extra_images)
    )

# Store paths relative to the project root.
df["image_path"] = df["image_id"].map(
    lambda image_id: os.path.relpath(
        image_lookup[image_id], BASE_DIR
    )
)


# --------------------------------------------------
# 5. MAP THE SEVEN DIAGNOSIS CLASSES
# --------------------------------------------------

print("\n========== LABEL MAPPING ==========")

class_names = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

unknown_labels = set(df["dx"]) - set(class_names)

if unknown_labels:
    raise ValueError(
        f"Unexpected diagnosis labels: {unknown_labels}"
    )

df["label"] = df["dx"].map(class_names)

with open(
    os.path.join(PROCESSED_DIR, "label_mapping.json"),
    "w"
) as file:
    json.dump(class_names, file, indent=4)

print(class_names)


# --------------------------------------------------
# 6. ORIGINAL CLASS DISTRIBUTION
# --------------------------------------------------

print("\n========== ORIGINAL CLASS DISTRIBUTION ==========")

original_counts = df["dx"].value_counts().sort_index()

print(original_counts)

imbalance_ratio = (
    original_counts.max() / original_counts.min()
)

print(
    "\nOriginal imbalance ratio (largest/smallest class):",
    round(imbalance_ratio, 2)
)


# --------------------------------------------------
# 7. LESION-LEVEL TRAIN / GLOBAL TEST SPLIT
# --------------------------------------------------

# --------------------------------------------------
# 7. LESION-LEVEL TRAIN / GLOBAL TEST SPLIT
# --------------------------------------------------

print("\n========== LESION-LEVEL SPLITTING ==========")

# Each lesion must have only one diagnosis label.
# We already verified this earlier.

# Create one record per unique lesion.
lesion_df = (
    df[["lesion_id", "dx"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Stratify at lesion level, not image level.
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_lesion_indices, test_lesion_indices = next(
    splitter.split(
        lesion_df[["lesion_id"]],
        lesion_df["dx"]
    )
)

# Obtain lesion IDs for each split.
train_lesions = set(
    lesion_df.iloc[train_lesion_indices]["lesion_id"]
)

test_lesions = set(
    lesion_df.iloc[test_lesion_indices]["lesion_id"]
)

# Assign every image of a lesion to the same split.
train_df = df[
    df["lesion_id"].isin(train_lesions)
].copy()

test_df = df[
    df["lesion_id"].isin(test_lesions)
].copy()

# Verify no lesion-level leakage.
overlap = train_lesions.intersection(test_lesions)

print("Total images:", len(df))
print("Training images:", len(train_df))
print("Global test images:", len(test_df))

print("Training lesions:", len(train_lesions))
print("Test lesions:", len(test_lesions))

print("Overlapping lesions:", len(overlap))

if overlap:
    raise RuntimeError(
        "LEAKAGE DETECTED: Lesions overlap between splits!"
    )

if set(train_df["image_id"]).intersection(
    set(test_df["image_id"])
):
    raise RuntimeError(
        "IMAGE LEAKAGE DETECTED!"
    )

if set(train_df["lesion_id"]) | set(test_df["lesion_id"]) != set(df["lesion_id"]):
    raise RuntimeError(
        "Some lesions were not assigned to either split."
    )

print("\nSUCCESS: No lesion-level data leakage detected.")


# --------------------------------------------------
# 8. SAVE CLEANED DATA AND SPLITS
# --------------------------------------------------

df.to_csv(
    os.path.join(PROCESSED_DIR, "cleaned_metadata.csv"),
    index=False
)

train_df.to_csv(
    os.path.join(PROCESSED_DIR, "train.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(PROCESSED_DIR, "global_test.csv"),
    index=False
)

print("\nSaved cleaned metadata and split CSV files.")


# --------------------------------------------------
# 9. CLASS DISTRIBUTION AFTER SPLITTING
# --------------------------------------------------

print("\n========== CLASS DISTRIBUTION BY SPLIT ==========")

all_classes = sorted(class_names.keys())

distribution = pd.DataFrame({
    "class": all_classes,
    "original": [
        (df["dx"] == c).sum() for c in all_classes
    ],
    "train": [
        (train_df["dx"] == c).sum() for c in all_classes
    ],
    "global_test": [
        (test_df["dx"] == c).sum() for c in all_classes
    ]
})

print(distribution.to_string(index=False))

distribution.to_csv(
    os.path.join(PROCESSED_DIR, "class_distribution.csv"),
    index=False
)


# --------------------------------------------------
# 10. IMBALANCE RATIOS
# --------------------------------------------------

def calculate_imbalance(counts):

    nonzero_counts = counts[counts > 0]

    if len(nonzero_counts) == 0:
        return float("nan")

    return (
        nonzero_counts.max() / nonzero_counts.min()
    )


for split_name, split_df in [
    ("Original", df),
    ("Training", train_df),
    ("Global Test", test_df)
]:

    counts = split_df["dx"].value_counts()

    print(
        f"{split_name} imbalance ratio:",
        round(calculate_imbalance(counts), 2)
    )


# --------------------------------------------------
# 11. VISUALIZE ORIGINAL CLASS DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.barplot(
    x=original_counts.index,
    y=original_counts.values
)

plt.title("HAM10000 Original Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "class_distribution.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. VISUALIZE TRAIN / TEST DISTRIBUTION
# --------------------------------------------------

plot_data = distribution.melt(
    id_vars="class",
    value_vars=["train", "global_test"],
    var_name="split",
    value_name="images"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_data,
    x="class",
    y="images",
    hue="split"
)

plt.title("Training vs Global Test Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "split_distribution.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n========== M1 PREPARATION SUMMARY ==========")

print("Total images:", len(df))
print("Training images:", len(train_df))
print("Global test images:", len(test_df))
print("Disease classes:", df["dx"].nunique())
print("Unique lesions:", df["lesion_id"].nunique())
print("Lesion overlap:", len(overlap))


