"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN

"""
# data_handling.py

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from image_processing import preprocess_image  # Import from image_processing module

def image_generator(dataset_path, batch_size, target_size=(256, 256)):
    datagen = ImageDataGenerator(rescale=1./127.5, preprocessing_function=lambda x: x - 1.0)
    return datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,  # No labels are needed
        shuffle=True)  # Shuffle the images

def visualize_model(model, filename='model_architecture.png'):
    """
    Generate an image of the model architecture.

    Parameters:
    model (tf.keras.Model): The Keras model to be visualized.
    filename (str): The name of the file where to save the model architecture image.
    """
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved as {filename}")
