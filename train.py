"""
train.py
--------
Training script.  Run from the command line:
    python train.py --dataset ./dataset --epochs 10

Saves the trained model to  saved_model/mango_model.keras
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset_loader import get_data_generators
from model import build_model


def train(dataset_dir, epochs=10, save_dir="saved_model"):
    """Train the model and save weights."""

    # 1. Load data
    train_gen, val_gen, class_names = get_data_generators(dataset_dir)

    # 2. Build model
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)

    # 3. Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "mango_model.keras")

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # 4. Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, checkpoint],
    )

    # 5. Save class names alongside the model
    class_names_path = os.path.join(save_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))

    print(f"\n[INFO] Model saved to      : {model_path}")
    print(f"[INFO] Class names saved to: {class_names_path}")

    return history


# ---- CLI entry point ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mango Disease Detector")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    args = parser.parse_args()

    train(args.dataset, args.epochs)
