import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Build transfer learning model using MobileNetV2
    """

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model for feature extraction
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model