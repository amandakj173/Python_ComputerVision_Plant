# DATA WRANGLING

# Import packages
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
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


# IMAGE PROCESSING

def processing(file_path):
	img = cv2.imread(file_path)  # Load image from path
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR (open CV convention)
	img = cv2.resize(img, (224, 224))  # Resize to standard image conventions
	img = img/255.0  # Normalise pizel values to range 0.0-1.0
	return img

process_train = np.array([processing(fp) for fp in train_df["file_path"]])  # Apply to training df
process_test = np.array([processing(fp) for fp in test_df["file_path"]])  # Apply to testing df


# OBJECT DETECTION

# Encode target labels
train_label = train_df["label"].values

label_encoder = LabelEncoder()
train_label = label_encoder.fit_transform(train_label)

class_names = label_encoder.classes_
print(class_names)