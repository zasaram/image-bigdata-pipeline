import os
import tensorflow as tf
from data_utils import load_data
from model_utils import build_model

# Import configuration
import config_example as config


def train_pipeline():
    """
    Main training pipeline for large-scale image processing
    """

    print("Loading dataset...")
    train_data, val_data = load_data(
        config.DATA_DIR,
        config.IMAGE_SIZE,
        config.BATCH_SIZE
    )

    num_classes = train_data.num_classes
    print("Number of classes:", num_classes)

    print("Building model...")
    model = build_model(num_classes, input_shape=(*config.IMAGE_SIZE, 3))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.EPOCHS
    )

    print("Saving model...")
    model.save(config.MODEL_PATH)

    print("Training completed successfully!")


if __name__ == "__main__":
    train_pipeline()