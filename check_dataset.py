import os
import pandas as pd
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\Major Project\HAM10000"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

METADATA_PATH = os.path.join(RAW_DIR, "HAM10000_metadata.csv")
IMAGE_DIR_1 = os.path.join(RAW_DIR, "HAM10000_images_part_1")
IMAGE_DIR_2 = os.path.join(RAW_DIR, "HAM10000_images_part_2")

# ============================================================
# START
# ============================================================

print("=" * 60)
print("HAM10000 DATASET VERIFICATION")
print("=" * 60)

# ============================================================
# 1. METADATA
# ============================================================

print("\n1. METADATA CHECK")

df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\n2. CLASS DISTRIBUTION")

class_counts = df["dx"].value_counts()

print(class_counts)

print("\nClass percentages:")

class_percentages = (
    df["dx"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(class_percentages)

# ============================================================
# 3. LESION INFORMATION
# ============================================================

print("\n3. LESION INFORMATION")

unique_images = df["image_id"].nunique()
unique_lesions = df["lesion_id"].nunique()

print("Unique image IDs:", unique_images)
print("Unique lesion IDs:", unique_lesions)

print(
    "Average images per lesion:",
    round(len(df) / unique_lesions, 2)
)

# ============================================================
# 4. DUPLICATE IMAGE IDs
# ============================================================

print("\n4. DUPLICATE IMAGE IDs")

duplicate_images = df["image_id"].duplicated().sum()

print("Duplicate image_id rows:", duplicate_images)

# ============================================================
# 5. MULTIPLE IMAGES PER LESION
# ============================================================

print("\n5. MULTIPLE IMAGES PER LESION")

lesion_counts = df.groupby("lesion_id")["image_id"].count()

multiple_image_lesions = (lesion_counts > 1).sum()

print(
    "Lesions with more than one image:",
    multiple_image_lesions
)

# ============================================================
# 6. IMAGE FILE CHECK
# ============================================================

print("\n6. IMAGE FILE CHECK")

image_files = {}

for folder in [IMAGE_DIR_1, IMAGE_DIR_2]:

    print("\nChecking:", folder)

    if not os.path.exists(folder):
        print("WARNING: Folder does not exist!")
        continue

    count = 0

    for filename in os.listdir(folder):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_id = os.path.splitext(filename)[0]

            image_files[image_id] = os.path.join(
                folder,
                filename
            )

            count += 1

    print("Images found:", count)

print("\nTotal unique image files:", len(image_files))

# ============================================================
# 7. MISSING / EXTRA IMAGES
# ============================================================

print("\n7. MISSING / EXTRA IMAGES")

metadata_ids = set(df["image_id"])
actual_ids = set(image_files.keys())

missing_images = metadata_ids - actual_ids
extra_images = actual_ids - metadata_ids

print(
    "Images in metadata but missing:",
    len(missing_images)
)

print(
    "Images in folders but not in metadata:",
    len(extra_images)
)

# ============================================================
# 8. IMAGE OPEN TEST + DIMENSIONS
# ============================================================

print("\n8. IMAGE VALIDITY CHECK")

dimensions = {}

checked = 0
failed = 0

for image_id, image_path in image_files.items():

    try:

        with Image.open(image_path) as img:

            size = img.size

            dimensions[size] = dimensions.get(size, 0) + 1

            checked += 1

    except Exception:

        failed += 1

print("Images successfully opened:", checked)
print("Images that failed to open:", failed)

print("\nImage dimensions:")

for size, count in sorted(
    dimensions.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{size[0]} x {size[1]} : {count} images"
    )

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Metadata rows:", len(df))
print("Image files:", len(image_files))
print("Classes:", df["dx"].nunique())
print("Unique lesions:", unique_lesions)
print("Duplicate image IDs:", duplicate_images)
print("Missing images:", len(missing_images))
print("Extra images:", len(extra_images))
print("Images successfully opened:", checked)
print("Images failed to open:", failed)

print("=" * 60)
print("DATASET VERIFICATION COMPLETED")
print("=" * 60)