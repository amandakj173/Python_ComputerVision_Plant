# DATA WRANGLING

# Import packages
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import(Layer, Input, RandomFlip, RandomRotation, RandomZoom, Conv2, MaxPooling2D, Flatten, Dense)

# Import data
base_dir = Path("plant_dataset")  # Set pathway where all image files stored
train_dir = base_dir / "train_ds"
test_dir = base_dir / "test_ds"

extensions = ["*.jpg", "*.jpeg", "*.png"]  # Set allowed image types

# Import train dataset
train_ds = []
categories = ["almond_plant", "cherry_plant", "maize_plant"]

for category in categories:
    folder_path = train_dir / category
    for ext in extensions:
        for file_path in folder_path.glob(ext):
            train_ds.append({
            "file_path": str(file_path),
            "label": category
            })

train_df = pd.DataFrame(train_ds)

# Import test dataset
test_ds = [str(p) for p in test_dir.iterdir() if p.is_file()]
test_df = pd.DataFrame({"file_path":test_ds})

# Check imported data
print("Train Samples:", train_df["label"].value_counts())
print("Test Samples:", len(test_df))