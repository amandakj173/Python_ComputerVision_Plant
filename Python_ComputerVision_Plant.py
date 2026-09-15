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