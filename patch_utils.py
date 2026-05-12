"""
patch_utils.py
--------------
Patch-based inference with severity-aware weighting
"""

import numpy as np
from PIL import Image


PATCH_SIZE = 128
MODEL_INPUT_SIZE = (224, 224)


def extract_patches(image: Image.Image, patch_size=PATCH_SIZE):
    img_array = np.array(image)
    h, w, c = img_array.shape
    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img_array[y : y + patch_size, x : x + patch_size]

            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, c), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            patches.append(patch)

    return patches


def preprocess_patch(patch_array, target_size=MODEL_INPUT_SIZE):
    img = Image.fromarray(patch_array)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def patch_based_predict(model, image: Image.Image, class_names: list):
    patches = extract_patches(image)

    all_preds = []

    for patch in patches:
        preprocessed = preprocess_patch(patch)
        pred = model.predict(preprocessed, verbose=0)
        all_preds.append(pred[0])

    all_preds = np.array(all_preds)  # (N, num_classes)

    # 🔥 Severity-based weighting
    healthy_idx = class_names.index("Healthy")

    weights = []
    for p in all_preds:
        healthy_prob = p[healthy_idx]

        # Severity = how non-healthy the patch is
        severity = 1 - healthy_prob

        # Avoid zero weight
        weight = severity + 0.05
        weights.append(weight)

    weights = np.array(weights)

    # 🔥 Special rule: Healthy only if ALL patches are healthy
    healthy_flags = [p[healthy_idx] > 0.7 for p in all_preds]

    if all(healthy_flags):
        final_preds = np.mean(all_preds, axis=0)
    else:
        final_preds = np.average(all_preds, axis=0, weights=weights)

    # Final class
    predicted_idx = int(np.argmax(final_preds))
    predicted_class = class_names[predicted_idx]

    probabilities = {
        name: round(float(final_preds[i]) * 100, 2)
        for i, name in enumerate(class_names)
    }

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "num_patches": len(patches),
    }