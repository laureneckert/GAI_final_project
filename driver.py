"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN
"""
# driver.py

import driverFunctions as df
import MLmodel

def main():
    """
    Main function to run the CycleGAN model for style transfer.
    """
    # Parameters
    epochs = 10
    batch_size = 1
    model_save_name = 'art_style_generator'
    train_model = True

    if train_model:
        # Load datasets and train the model
        art_images, photo_images = df.load_datasets()
        generator, discriminator = MLmodel.build_generator(), MLmodel.build_discriminator()
        MLmodel.compile_models(generator, discriminator)
        MLmodel.train_model(art_images, photo_images, generator, discriminator, epochs, batch_size)
        MLmodel.save_model(generator, model_save_name)
    else:
        # Load a pre-trained model
        generator = MLmodel.load_model(model_save_name)
        if generator is None:
            print("Model not found. Exiting.")
            return

    # Style transfer
    input_image_path = input("Enter the path of your photograph: ")
    styled_image = df.apply_art_style(input_image_path, generator)

    # Output
    df.save_or_display_image(styled_image, save=False, display=True)

if __name__ == "__main__":
    main()
