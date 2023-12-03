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

import image_processing as ip
import data_handling as dh
import MLmodel

def main():
    epochs = 10
    batch_size = 1
    model_save_name = 'art_style_generator'
    train_model_flag = True
    num_samples_imagenet = 1000  # Number of samples from ImageNet

    path_to_artbench_dataset = r"C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\artbench-10-python"

    if train_model_flag:
        # Load datasets
        art_images_gen = dh.image_generator(path_to_artbench_dataset, batch_size)
        photo_images_gen = dh.load_imagenet_subset(batch_size, num_samples_imagenet)

        # Initialize and compile models
        generator_AtoB, generator_BtoA = MLmodel.build_generator(), MLmodel.build_generator()
        discriminator_A, discriminator_B = MLmodel.build_discriminator(), MLmodel.build_discriminator()

        MLmodel.compile_models(generator_AtoB, discriminator_A)
        MLmodel.compile_models(generator_BtoA, discriminator_B)

        # Train the CycleGAN model
        MLmodel.train_model(art_images_gen, photo_images_gen, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, epochs, steps_per_epoch=100)

        # Save the trained models
        model_names = ['_AtoB', '_BtoA', '_DiscriminatorA', '_DiscriminatorB']
        for model, name in zip([generator_AtoB, generator_BtoA, discriminator_A, discriminator_B], model_names):
            MLmodel.save_model(model, model_save_name + name)
    else:
        # Load pre-trained models
        generator_AtoB = MLmodel.load_model(model_save_name + '_AtoB')
        generator_BtoA = MLmodel.load_model(model_save_name + '_BtoA')
        discriminator_A = MLmodel.load_model(model_save_name + '_DiscriminatorA')
        discriminator_B = MLmodel.load_model(model_save_name + '_DiscriminatorB')
        
        if not all([generator_AtoB, generator_BtoA, discriminator_A, discriminator_B]):
            print("One or more models not found. Exiting.")
            return
          
    # visualize model architectures for both generators and discriminators
    dh.visualize_model(generator_AtoB, filename='generator_AtoB_architecture.png')
    dh.visualize_model(generator_BtoA, filename='generator_BtoA_architecture.png')
    dh.visualize_model(discriminator_A, filename='discriminator_A_architecture.png')
    dh.visualize_model(discriminator_B, filename='discriminator_B_architecture.png')

    # Style transfer
    input_image_path = input("Enter the path of your photograph: ")
    styled_image = ip.apply_art_style(input_image_path, generator_AtoB)  # Choose the appropriate generator

    # Output
    ip.save_or_display_image(styled_image, save=False, display=True)

if __name__ == "__main__":
    main()
