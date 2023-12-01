"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN
"""
#driver.py

import driverFunctions as df
import MLmodel

def main():
    # Define parameters
    epochs = 10  # Number of training epochs (adjust as needed)
    batch_size = 1  # Batch size for training (adjust as needed)
    model_save_name = 'art_style_generator'  # Name to save your trained generator model

    # Load or train your model
    # Uncomment the following lines if you need to train the model
    # art_images, photo_images = df.load_datasets()  # Implement this function in driverFunctions.py
    # generator, discriminator = MLmodel.build_generator(), MLmodel.build_discriminator()
    # MLmodel.compile_models(generator, discriminator)
    # MLmodel.train_model(art_images, photo_images, generator, discriminator, epochs, batch_size)
    # MLmodel.save_model(generator, model_save_name)

    # Or load the pre-trained model
    generator = MLmodel.load_model(model_save_name)
    if generator is None:
        print("Model not found. Exiting.")
        return

    # Input from user for style transfer
    input_image_path = input("Enter the path of your photograph: ")

    # Apply art style using the loaded generator
    styled_image = df.apply_art_style(input_image_path, generator)

    # Save or display styled image
    df.save_or_display_image(styled_image, save=True, display=True)
if __name__ == "__main__":
    main()
