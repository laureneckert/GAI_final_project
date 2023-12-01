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

def train_model(art_images, photo_images, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, batch_size):
    # Define loss functions
    adversarial_loss = BinaryCrossentropy(from_logits=True)
    cycle_loss = MeanAbsoluteError()
    identity_loss = MeanAbsoluteError()

    # Define the optimizers
    gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    for epoch in range(epochs):
        for i in range(min(len(art_images), len(photo_images)) // batch_size):
            # Sample a batch of images from both domains
            ...

            with tf.GradientTape(persistent=True) as tape:
                # Forward cycle (Photo to Art to Photo)
                fake_art_images = generator_AtoB(batch_photo_images)
                cycled_photo_images = generator_BtoA(fake_art_images)

                # Backward cycle (Art to Photo to Art)
                fake_photo_images = generator_BtoA(batch_art_images)
                cycled_art_images = generator_AtoB(fake_photo_images)

                # Identity mapping (optional)
                same_art_images = generator_AtoB(batch_art_images)
                same_photo_images = generator_BtoA(batch_photo_images)

                # Discriminator output
                ...

                # Calculate losses
                total_cycle_loss = cycle_loss(batch_art_images, cycled_art_images) + cycle_loss(batch_photo_images, cycled_photo_images)
                total_identity_loss = identity_loss(batch_art_images, same_art_images) + identity_loss(batch_photo_images, same_photo_images)
                total_gen_AtoB_loss = adversarial_loss(tf.ones_like(disc_fake_art), disc_fake_art) + total_cycle_loss + total_identity_loss
                total_gen_BtoA_loss = adversarial_loss(tf.ones_like(disc_fake_photo), disc_fake_photo) + total_cycle_loss + total_identity_loss
                ...

                # Update the weights
                ...

    return generator_AtoB, generator_BtoA


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
