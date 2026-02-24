import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, image_size, batch_size):
    """
    Load images from directory.
    Expected structure:
    data/
        class1/
        class2/
        ...
    """

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data