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
from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, LeakyReLU, Add, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanAbsoluteError

def resnet_block(input_layer, filters, kernel_size=(3, 3)):
    """
    Create a ResNet block with two convolutional layers and a skip connection.

    Parameters:
    input_layer (Tensor): Input tensor to the ResNet block.
    filters (int): Number of filters in the convolutional layers.
    kernel_size (tuple): Size of the kernel for the convolutional layers.

    Returns:
    Tensor: Output tensor of the ResNet block.
    """
    x = Conv2D(filters, kernel_size, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Skip Connection
    x = Add()([x, input_layer])
    return Activation('relu')(x)

def build_generator(input_shape=(256, 256, 3), num_resnet_blocks=9):
    """
    Build a generator model using a ResNet-based architecture.

    Parameters:
    input_shape (tuple): Shape of the input image.
    num_resnet_blocks (int): Number of ResNet blocks.

    Returns:
    Model: Generator model.
    """
    inputs = Input(shape=input_shape)

    # Initial Convolution Block
    x = Conv2D(64, (7, 7), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Downsampling
    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ResNet Blocks
    for _ in range(num_resnet_blocks):
        x = resnet_block(x, 256)

    # Upsampling
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output Convolution
    x = Conv2D(3, (7, 7), padding='same')(x)
    x = Activation('tanh')(x)

    return Model(inputs, x, name='Generator')

def build_discriminator(input_shape=(256, 256, 3)):
    """
    Build a discriminator model using a PatchGAN architecture.

    Parameters:
    input_shape (tuple): Shape of the input image.

    Returns:
    Model: Discriminator model.
    """
    inputs = Input(shape=input_shape)

    # Convolutional layers with increasing filters
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Output convolution layer
    x = Conv2D(1, (4, 4), padding='same')(x)

    return Model(inputs, x, name='Discriminator')

def compile_models(generator, discriminator, lr=0.0002, beta_1=0.5):
    """
    Compile the generator and discriminator models with specified optimizers and loss functions.

    Parameters:
    generator (Model): Generator model to be compiled.
    discriminator (Model): Discriminator model to be compiled.
    lr (float): Learning rate for the Adam optimizer.
    beta_1 (float): Beta1 parameter for the Adam optimizer.
    """
    generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=beta_1))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=beta_1))

def train_model(art_images, photo_images, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, batch_size):
    """
    Train the CycleGAN models on the provided datasets.

    Parameters:
    art_images (Iterator): Batched images from the ArtBench-10 dataset.
    photo_images (Iterator): Batched images from the ImageNet dataset.
    generator_AtoB (Model): Generator model to translate from domain A to B.
    generator_BtoA (Model): Generator model to translate from domain B to A.
    discriminator_A (Model): Discriminator model for domain A.
    discriminator_B (Model): Discriminator model for domain B.
    epochs (int): Number of epochs for training.
    batch_size (int): Size of the image batches.
    """
    # Loss functions
    adversarial_loss = BinaryCrossentropy(from_logits=True)
    cycle_loss = MeanAbsoluteError()
    identity_loss = MeanAbsoluteError()

    # Optimizers
    gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    # Training loop
    for epoch in range(epochs):
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
    Save a trained model to the specified directory.

    Parameters:
    model (Model): The trained model to be saved.
    model_name (str): Name for the saved model file.
    save_dir (str): Directory to save the model.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f"Model saved at {model_path}")

def load_model(model_name, model_dir='models'):
    """
    Load a trained model from the specified directory.

    Parameters:
    model_name (str): Name of the model file to be loaded.
    model_dir (str): Directory from where to load the model.

    Returns:
    Model: The loaded Keras model.
    """
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"No model found at {model_path}")
        return None
