"""The 2022 ConvNet as a Keras 3 model.

Only ``keras.*`` APIs are used, so the same model runs on the TensorFlow,
JAX and PyTorch backends (set ``KERAS_BACKEND`` before importing keras).
"""

from __future__ import annotations

import keras
from keras import layers

from .config import TrainConfig


def build_keras_model(cfg: TrainConfig) -> keras.Model:
    """Build and compile the 2022 architecture.

    Three convolution blocks (11x11, 5x5, then two 1x1 convolutions, each
    followed by batch normalisation and max pooling) feed two 1024-unit
    dense layers with 20% dropout and a single sigmoid output: the
    probability that the page is moving. Compiled with Adam at
    ``cfg.learning_rate``, binary cross-entropy and accuracy.

    ``cfg.seed`` seeds the weight initialisation and the dropout masks.
    """
    keras.utils.set_random_seed(cfg.seed)
    model = keras.Sequential(
        [
            keras.Input(shape=cfg.input_shape),
            layers.Conv2D(
                32,
                (11, 11),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (5, 5), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((3, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (1, 1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (1, 1), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="doc_motion",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
