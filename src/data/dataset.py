
# src/data/dataset.py

import os
import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)


# --------------------------------------------------
# 2. IMAGE TRANSFORMATIONS
# --------------------------------------------------

# Training pipeline: augmentation is applied only here.

def get_train_transforms():

    return A.Compose([

        A.Resize(height=224, width=224),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.Rotate(limit=15, p=0.5),

        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
            p=0.3
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()

    ])


# Test pipeline: deterministic transformations only.

def get_test_transforms():

    return A.Compose([

        A.Resize(height=224, width=224),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()

    ])


# --------------------------------------------------
# 3. HAM10000 PYTORCH DATASET
# --------------------------------------------------

class HAM10000Dataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        required_columns = ["image_path", "label"]

        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(
                    f"Missing column in CSV: {column}"
                )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join(
            BASE_DIR,
            row["image_path"]
        )

        # Read image using OpenCV.
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        # OpenCV loads BGR. Convert it to RGB.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = int(row["label"])

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# --------------------------------------------------
# 4. CREATE TRAIN AND GLOBAL TEST DATASETS
# --------------------------------------------------

def create_dataloaders(batch_size=32):

    train_csv = os.path.join(
        PROCESSED_DIR, "train.csv"
    )

    test_csv = os.path.join(
        PROCESSED_DIR, "global_test.csv"
    )

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            "train.csv not found. Run prepare_data.py first."
        )

    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            "global_test.csv not found. Run prepare_data.py first."
        )

    train_dataset = HAM10000Dataset(
        csv_file=train_csv,
        transform=get_train_transforms()
    )

    test_dataset = HAM10000Dataset(
        csv_file=test_csv,
        transform=get_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


# --------------------------------------------------
# 5. TEST THE PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    train_loader, test_loader = create_dataloaders(
        batch_size=32
    )

    print("\n========== DATALOADER VERIFICATION ==========")

    print("Training batches:", len(train_loader))
    print("Global test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("\nTraining image batch shape:", images.shape)
    print("Training label batch shape:", labels.shape)

    test_images, test_labels = next(iter(test_loader))

    print("\nTest image batch shape:", test_images.shape)
    print("Test label batch shape:", test_labels.shape)

    print("\nPreprocessing pipeline working successfully!")
