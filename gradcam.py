"""
gradcam.py
----------
Grad-CAM implementation for visual explainability.
Generates a heatmap highlighting regions the model focuses on,
then overlays it on the original image.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generate a Grad-CAM heatmap for the predicted class.

    Args:
        model                : Keras Model
        img_array            : np.ndarray, shape (1, 224, 224, 3), float32 [0,1]
        last_conv_layer_name : str – name of the last conv layer.
                               If None, auto-detected from model.

    Returns:
        heatmap : np.ndarray (H, W) float32 in [0, 1]
    """

    # Auto-detect last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # conv-like layer
                last_conv_layer_name = layer.name
                break

    # Build a sub-model: input → [last_conv_output, predictions]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # Forward pass + record gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_class]

    # Gradients of the predicted class w.r.t. the conv outputs
    grads = tape.gradient(class_score, conv_outputs)

    # Global-average-pool the gradients → channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv feature maps by the importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(original_image: Image.Image, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        original_image : PIL.Image (RGB)
        heatmap        : np.ndarray (H, W) float32 in [0, 1]
        alpha          : float – blending weight for the heatmap

    Returns:
        PIL.Image (RGB) – the blended result
    """
    # Resize heatmap to match original image
    img = np.array(original_image)
    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to colormap (JET)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    # Blend
    blended = np.uint8(alpha * colormap + (1 - alpha) * img)
    return Image.fromarray(blended)


def generate_gradcam(model, image: Image.Image):
    """
    End-to-end Grad-CAM: preprocess image → heatmap → overlay.

    Args:
        model : Keras Model
        image : PIL.Image (RGB)

    Returns:
        PIL.Image – original image with Grad-CAM overlay
    """
    # Preprocess for model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(model, img_array)

    # Overlay on the ORIGINAL (full-size) image
    result = overlay_heatmap(image, heatmap)
    return result
