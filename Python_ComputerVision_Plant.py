# DATA WRANGLING

# Import packages
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import(Layer, Input, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, Conv2D, MaxPooling2D, Flatten, Dense)

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

# Shuffle Train Images & Labels
process_train, train_label = shuffle(
    process_train, train_label, random_state = 42
)

# Image Augmentation
image_aug = Sequential([
    Input(shape=(224, 224, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(height_factor = 0.2, width_factor = 0.2),
    RandomBrightness(factor = 0.2),
], name = "image_aug")


# OBJECT CLASSIFICATION

# CNN Architecture
cnn_model = Sequential([
    image_aug,  # Use augmented image dataset
    Rescaling(1./255)  # Normalise 0-255 inputs
    Conv2D(32, (3,3), activation = "relu"),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation = "relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation = "relu"),
    Dense(len(class_names), activation = "softmax")
])

# Summarise Architecture
cnn_model.summary()

# Compile Model
cnn_model.compile(
    optimizer = "adam",  # Dynamically adjust learning rate during training
    loss = "sparse_categorical_crossentropy",  # Loss function used when targets integer-coded labels
    metrics = ["accuracy"]  # Measure % correctly classified image during training & validation
)


# Train Model
history = cnn_model.fit(
    process_train,  # Uses training images
    train_label,  # Uses label-encoded class targets
    epochs = 15,  # Performs 15 full passes over dataset w. data_aug generating new random tranformations each time
    batch_size = 4  # Give 5-6 steps per epoch
    validation_split = 0.3,  # Hold out 30% data for validation
    verbose = 2
)

