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
from keras.layers import Input, Conv2D, Activation, LeakyReLU, Add, BatchNormalization, UpSampling2D
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

def train_model(art_images_gen, photo_images_gen, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, steps_per_epoch):
    """
    Train CycleGAN models on provided datasets.

    Parameters:
    art_images_gen (Iterator): Generator for ArtBench-10 dataset images.
    photo_images_gen (Iterator): Generator for ImageNet dataset images.
    generator_AtoB (Model): Generator model from domain A (Photo) to B (Art).
    generator_BtoA (Model): Generator model from domain B (Art) to A (Photo).
    discriminator_A (Model): Discriminator model for domain A (Photo).
    discriminator_B (Model): Discriminator model for domain B (Art).
    epochs (int): Number of training epochs.
    steps_per_epoch (int): Number of steps per epoch.
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
        print(f"Epoch {epoch+1}/{epochs}")

        for step in range(steps_per_epoch):
            # Get a batch of images from both domains
            art_images = next(art_images_gen)
            photo_images = next(photo_images_gen)

            with tf.GradientTape(persistent=True) as tape:
                # Forward cycle: Photo -> Art -> Photo
                fake_art_images = generator_AtoB(photo_images, training=True)
                cycled_photo_images = generator_BtoA(fake_art_images, training=True)

                # Backward cycle: Art -> Photo -> Art
                fake_photo_images = generator_BtoA(art_images, training=True)
                cycled_art_images = generator_AtoB(fake_photo_images, training=True)

                # Discriminator outputs
                disc_real_photo = discriminator_A(photo_images, training=True)
                disc_fake_photo = discriminator_A(fake_photo_images, training=True)
                disc_real_art = discriminator_B(art_images, training=True)
                disc_fake_art = discriminator_B(fake_art_images, training=True)

                # Generator adversarial loss
                gen_AtoB_loss = adversarial_loss(tf.ones_like(disc_fake_art), disc_fake_art)
                gen_BtoA_loss = adversarial_loss(tf.ones_like(disc_fake_photo), disc_fake_photo)

                # Total generator loss
                total_gen_AtoB_loss = gen_AtoB_loss + cycle_loss(photo_images, cycled_photo_images) + identity_loss(art_images, fake_art_images)
                total_gen_BtoA_loss = gen_BtoA_loss + cycle_loss(art_images, cycled_art_images) + identity_loss(photo_images, fake_photo_images)

                # Discriminator loss
                disc_A_loss = adversarial_loss(tf.ones_like(disc_real_photo), disc_real_photo) + adversarial_loss(tf.zeros_like(disc_fake_photo), disc_fake_photo)
                disc_B_loss = adversarial_loss(tf.ones_like(disc_real_art), disc_real_art) + adversarial_loss(tf.zeros_like(disc_fake_art), disc_fake_art)

            # Calculate gradients and update model weights
            generator_AtoB_gradients = tape.gradient(total_gen_AtoB_loss, generator_AtoB.trainable_variables)
            generator_BtoA_gradients = tape.gradient(total_gen_BtoA_loss, generator_BtoA.trainable_variables)
            discriminator_A_gradients = tape.gradient(disc_A_loss, discriminator_A.trainable_variables)
            discriminator_B_gradients = tape.gradient(disc_B_loss, discriminator_B.trainable_variables)

            gen_optimizer.apply_gradients(zip(generator_AtoB_gradients, generator_AtoB.trainable_variables))
            gen_optimizer.apply_gradients(zip(generator_BtoA_gradients, generator_BtoA.trainable_variables))
            disc_optimizer.apply_gradients(zip(discriminator_A_gradients, discriminator_A.trainable_variables))
            disc_optimizer.apply_gradients(zip(discriminator_B_gradients, discriminator_B.trainable_variables))

        print(f"Completed Epoch {epoch+1}")

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
