"""
predict.py
----------
Convenience module that loads the saved model and runs patch-based prediction.
Can also be used as a standalone CLI tool:
    python predict.py --image test.jpg
"""

import argparse
import os

import numpy as np
from PIL import Image
import tensorflow as tf

from patch_utils import patch_based_predict


# --- Defaults ---
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "mango_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")


def load_model_and_classes(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    """Load the trained Keras model and class names from disk."""
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def predict_image(image_path, model=None, class_names=None):
    """
    Run patch-based prediction on a single image file.

    Args:
        image_path  : str – path to the input image
        model       : optional pre-loaded model
        class_names : optional pre-loaded class names

    Returns:
        dict with predicted_class, probabilities, num_patches
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()

    image = Image.open(image_path).convert("RGB")
    result = patch_based_predict(model, image, class_names)
    return result


# ---- CLI entry point ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mango disease (patch-based)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    result = predict_image(args.image)

    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Patches used    : {result['num_patches']}")
    print("Probabilities   :")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:20s} : {prob:.2f}%")
