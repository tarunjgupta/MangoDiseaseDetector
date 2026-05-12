"""
model.py
--------
Builds the transfer-learning model:
    MobileNetV2 (frozen base) + custom classification head.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# --- Constants ---
IMG_SIZE = (224, 224)
NUM_CLASSES = 4


def build_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Build a MobileNetV2-based classifier.

    Architecture:
        MobileNetV2 (ImageNet, frozen) →
        GlobalAveragePooling2D →
        Dense(128, relu) → Dropout(0.3) →
        Dense(num_classes, softmax)

    Returns:
        compiled Keras Model
    """

    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the base model weights
    base_model.trainable = False

    # Build custom head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model
