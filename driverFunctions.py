"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN

"""
# driverFunctions.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.utils import plot_model

def preprocess_image(image, target_size=(256, 256), normalize=True):
    """
    Preprocess an image by resizing and optionally normalizing it.

    Parameters:
    image (Tensor or Image): The image to preprocess.
    target_size (tuple): The target size for resizing.
    normalize (bool): Whether to normalize the image.

    Returns:
    Tensor: The preprocessed image tensor.
    """
    if not isinstance(image, tf.Tensor):
        image = img_to_array(image)

    image = tf.image.resize(image, target_size)

    if normalize:
        image = (image / 127.5) - 1

    return image


def postprocess_image(image_tensor):
    """
    Postprocess the image tensor to convert it into a displayable format.

    Parameters:
    image_tensor (numpy.ndarray): The image tensor output from the model.

    Returns:
    numpy.ndarray: Postprocessed image suitable for display or saving.
    """
    image_tensor = (image_tensor * 127.5) + 127.5
    image_tensor = np.array(image_tensor, dtype=np.uint8)
    if np.ndim(image_tensor) > 3:
        image_tensor = image_tensor[0]
    return image_tensor

def apply_art_style(image_path, generator):
    input_image = load_img(image_path)
    processed_image = preprocess_image(input_image, normalize=True)
    processed_image = np.expand_dims(processed_image, axis=0)
    generated_image = generator.predict(processed_image)
    return postprocess_image(generated_image)

def save_or_display_image(image, save=False, display=True, save_path='styled_image.jpg'):
    """
    Save or display the processed image.

    Parameters:
    image (numpy.ndarray): The image to be saved or displayed.
    save (bool): Whether to save the image to disk.
    display (bool): Whether to display the image.
    save_path (str): Path to save the image if save is True.
    """
    if save:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if display:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

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

def visualize_model(model, filename='model_architecture.png'):
    """
    Generate an image of the model architecture.

    Parameters:
    model (tf.keras.Model): The Keras model to be visualized.
    filename (str): The name of the file where to save the model architecture image.
    """
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved as {filename}")

def load_imagenet_subset(batch_size, num_samples, img_size=(256, 256), shuffle_buffer_size=10000):
    dataset = tfds.load('imagenet_v2', split='train', as_supervised=True)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.take(num_samples)
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, img_size))
    dataset = dataset.batch(batch_size)
    return dataset

