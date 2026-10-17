"""VGG16 model building and compilation for aircraft damage classification."""

from typing import Tuple

import keras
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam

from aircraft_ai import config
from aircraft_ai.logging_setup import setup_logger

logger = setup_logger("aircraft_ai.classifier")


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    weights: str | None = "imagenet",
    freeze_backbone: bool = True,
) -> Sequential:
    """Build the exact VGG16 feature extractor and binary classification head.
    
    Academic specification:
    1. Base: VGG16(weights=weights, include_top=False, input_shape=input_shape)
    2. Extract last layer output, apply Flatten(), wrap in Model(base.input, output)
    3. Freeze all base layers (trainable = False)
    4. Head: Sequential([
           base_model,
           Dense(512, activation='relu'),
           Dropout(0.3),
           Dense(512, activation='relu'),
           Dropout(0.3),
           Dense(1, activation='sigmoid')
       ])
    """
    logger.info("Building VGG16 base model (weights=%s, input_shape=%s)...", weights, input_shape)
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    output = base_model.layers[-1].output
    output = Flatten()(output)
    base_model = Model(base_model.input, output, name="vgg16_feature_extractor")

    if freeze_backbone:
        for layer in base_model.layers:
            layer.trainable = False
        logger.info("Froze all %d layers of VGG16 base model.", len(base_model.layers))

    model = Sequential(name="vgg16_aircraft_damage_classifier")
    model.add(base_model)
    model.add(Dense(512, activation="relu", name="dense_1"))
    model.add(Dropout(0.3, name="dropout_1"))
    model.add(Dense(512, activation="relu", name="dense_2"))
    model.add(Dropout(0.3, name="dropout_2"))
    model.add(Dense(1, activation="sigmoid", name="classification_head"))

    return model


def compile_classifier(
    model: Sequential,
    learning_rate: float = config.LEARNING_RATE,
) -> Sequential:
    """Compile model with Adam(learning_rate), binary_crossentropy loss, and accuracy metric."""
    logger.info("Compiling model with Adam(lr=%f), binary_crossentropy, accuracy...", learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_compiled_model(weights: str | None = "imagenet") -> Sequential:
    """Convenience factory function to build and compile the classifier."""
    model = build_vgg16_classifier(weights=weights)
    return compile_classifier(model)
