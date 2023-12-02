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
    epochs = 10
    batch_size = 1
    model_save_name = 'art_style_generator'
    train_model_flag = True  # Flag to control training or loading of model

    pathToArtBenchDataset = r"C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\artbench-10-python"
    pathToImageNetDataset = ""
    if train_model_flag:
        # Load datasets and train the model
        art_images_gen, photo_images_gen = df.load_datasets(pathToArtBenchDataset, pathToImageNetDataset, batch_size)
        generator_AtoB, generator_BtoA = MLmodel.build_generator(), MLmodel.build_generator()
        discriminator_A, discriminator_B = MLmodel.build_discriminator(), MLmodel.build_discriminator()

        MLmodel.compile_models(generator_AtoB, discriminator_A)
        MLmodel.compile_models(generator_BtoA, discriminator_B)

        MLmodel.train_model(art_images_gen, photo_images_gen, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, steps_per_epoch=100)
        MLmodel.save_model(generator_AtoB, model_save_name + '_AtoB')
        MLmodel.save_model(generator_BtoA, model_save_name + '_BtoA')
    else:
        # Load pre-trained model
        generator_AtoB = MLmodel.load_model(model_save_name + '_AtoB')
        generator_BtoA = MLmodel.load_model(model_save_name + '_BtoA')
        if generator_AtoB is None or generator_BtoA is None:
            print("Model(s) not found. Exiting.")
            return

    # Visualize model architectures
    df.visualize_model(generator_AtoB, filename='generator_AtoB_architecture.png')
    df.visualize_model(generator_BtoA, filename='generator_BtoA_architecture.png')

    # Style transfer
    input_image_path = input("Enter the path of your photograph: ")
    styled_image = df.apply_art_style(input_image_path, generator_AtoB)  # Choose the appropriate generator

    # Output
    df.save_or_display_image(styled_image, save=False, display=True)

if __name__ == "__main__":
    main()
