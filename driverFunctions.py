"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN

"""
#driverFunctions.py

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
import cv2

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = (image - 127.5) / 127.5  # Normalize the image
    return np.expand_dims(image, axis=0)

def postprocess_image(image_tensor):
    image_tensor = (image_tensor * 127.5) + 127.5
    image_tensor = np.array(image_tensor, dtype=np.uint8)
    if np.ndim(image_tensor) > 3:
        image_tensor = image_tensor[0]
    return image_tensor

def apply_art_style(image_path, generator):
    input_image = load_and_preprocess_image(image_path)
    generated_image = generator.predict(input_image)
    return postprocess_image(generated_image)

def save_or_display_image(image, save=False, display=False, save_path='styled_image.jpg'):
    if save:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if display:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
