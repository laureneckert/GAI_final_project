"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN
"""

# MLmodel.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
import os

def build_generator():
    """
    Build the generator model for the CycleGAN.
    This function should return a Keras Model.
    """
    # Sample generator structure - modify as per requirements
    inputs = Input(shape=(256, 256, 3))

    # Downsample
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    # Upsample
    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = Activation('tanh')(x)

    return Model(inputs, x, name='Generator')

def build_discriminator():
    """
    Build the discriminator model for the CycleGAN.
    This function should return a Keras Model.
    """
    # Sample discriminator structure - modify as per requirements
    inputs = Input(shape=(256, 256, 3))

    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs, x, name='Discriminator')

def compile_models(generator, discriminator, lr=0.0002, beta_1=0.5):
    """
    Compile the CycleGAN models.
    """
    # Compile the generator
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=beta_1))

    # For the discriminator, freeze generator's layers
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=beta_1))

def train_model(art_images, photo_images, generator, discriminator, epochs, batch_size):
    """
    Train the CycleGAN model.
    art_images: Art images from ArtBench-10 dataset
    photo_images: Photograph images from ImageNet dataset
    """
    for epoch in range(epochs):
        # Implement the training logic here
        pass

def save_model(model, model_name, save_dir='models'):
    """
    Save a trained model.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f"Model saved at {model_path}")

def load_model(model_name, model_dir='models'):
    """
    Load a trained model.
    """
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"No model found at {model_path}")
        return None

# Example usage:
# gen = build_generator()
# disc = build_discriminator()
# compile_models(gen, disc)
# train_model(art_images, photo_images, gen, disc, epochs=10, batch_size=1)
# save_model(gen, 'art_style_generator')
