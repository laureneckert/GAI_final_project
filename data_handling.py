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
    """
    Create a generator that yields batches of images from a dataset directory.

    Parameters:
    dataset_path (str): Path to the dataset directory.
    batch_size (int): Number of images per batch.
    target_size (tuple): Desired size for each image as (width, height).

    Returns:
    Iterator: A generator that yields image batches.
    """
    datagen = ImageDataGenerator(rescale=1./127.5, preprocessing_function=lambda x: x - 1.0)
    return datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None)  # 'None' means no labels are returned

def load_datasets(artbench_path, imagenet_path, batch_size, target_size=(256, 256)):
    """
    Create generators for the ArtBench-10 and ImageNet datasets.

    Parameters:
    artbench_path (str): Path to the ArtBench-10 dataset directory.
    imagenet_path (str): Path to the ImageNet dataset directory.
    batch_size (int): Number of images per batch.
    target_size (tuple): Desired size for each image as (width, height).

    Returns:
    Tuple[Iterator, Iterator]: Tuple of generators for both datasets.
    """
    art_images_generator = image_generator(artbench_path, batch_size, target_size)
    photo_images_generator = image_generator(imagenet_path, batch_size, target_size)

    return art_images_generator, photo_images_generator

def load_imagenet_subset(batch_size, num_samples, img_size=(256, 256), shuffle_buffer_size=10000):
    dataset = tfds.load('imagenet_v2', split='train', as_supervised=True)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.take(num_samples)
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, img_size))
    dataset = dataset.batch(batch_size)
    return dataset

def visualize_model(model, filename='model_architecture.png'):
    """
    Generate an image of the model architecture.

    Parameters:
    model (tf.keras.Model): The Keras model to be visualized.
    filename (str): The name of the file where to save the model architecture image.
    """
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved as {filename}")
