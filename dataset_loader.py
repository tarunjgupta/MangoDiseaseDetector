"""
dataset_loader.py
-----------------
Loads the mango disease dataset from a directory structure.
Expected layout:
    dataset/
        Anthracnose/
        Bacterial_Canker/
        Scab/
        Healthy/

Returns tf.data-compatible generators with data augmentation for training.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Constants ---
IMG_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 16
CLASS_NAMES = ["Anthracnose", "Bacterial_Canker", "Healthy", "Scab"]


def get_data_generators(dataset_dir, val_split=0.2, max_per_class=200):
    """
    Create training and validation data generators with augmentation.

    Args:
        dataset_dir : str  – path to root dataset folder
        val_split   : float – fraction reserved for validation
        max_per_class: int  – max images to use per class (not enforced by
                              generator, but documented for manual subsetting)

    Returns:
        (train_gen, val_gen, class_names)
    """

    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest",
        validation_split=val_split,
    )

    # Validation generator – only rescale, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    # Derive sorted class names from the generator
    class_indices = train_gen.class_indices          # {'Anthracnose': 0, ...}
    class_names = sorted(class_indices, key=class_indices.get)

    print(f"\n[INFO] Classes found : {class_names}")
    print(f"[INFO] Training samples  : {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}\n")

    return train_gen, val_gen, class_names
