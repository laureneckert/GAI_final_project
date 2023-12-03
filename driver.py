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
    epochs = 10
    batch_size = 1
    model_save_name = 'art_style_generator'
    train_model_flag = True
    num_samples_imagenet = 1000  # Number of samples from ImageNet

    path_to_artbench_dataset = r"C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\artbench-10-python"

    if train_model_flag:
        # Load ArtBench dataset
        art_images_gen = df.image_generator(path_to_artbench_dataset, batch_size)

        # Load ImageNet subset
        photo_images_gen = df.load_imagenet_subset(batch_size, num_samples_imagenet)

        # Initialize and compile models
        generator_AtoB, generator_BtoA = MLmodel.build_generator(), MLmodel.build_generator()
        discriminator_A, discriminator_B = MLmodel.build_discriminator(), MLmodel.build_discriminator()

        MLmodel.compile_models(generator_AtoB, discriminator_A)
        MLmodel.compile_models(generator_BtoA, discriminator_B)

        # Train the CycleGAN model
        MLmodel.train_model(art_images_gen, photo_images_gen, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, steps_per_epoch=100)

        # Save the trained models
        MLmodel.save_model(generator_AtoB, model_save_name + '_AtoB')
        MLmodel.save_model(generator_BtoA, model_save_name + '_BtoA')
    else:
        # Load pre-trained models
        generator_AtoB = MLmodel.load_model(model_save_name + '_AtoB')
        generator_BtoA = MLmodel.load_model(model_save_name + '_BtoA')
        if not generator_AtoB or not generator_BtoA:
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
