"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN

"""
#image_processing.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
#from keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import img_to_array, load_img


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
