"""
Lauren Eckert
Jaric Abadinas

Generative AI
Final Project
Project topic: transitive learning - applying art historical styles to personal photos
Databases: ArtBench-10, ImageNet
Model: CycleGAN

driver.py
"""

import driverFunctions as df
import MLmodel

def main():
    # Load trained CycleGAN model (modify with actual model loading code)
    generator = MLmodel.load_trained_model('path/to/trained_model')

    # Input from user
    input_image_path = input("Enter the path of your photograph: ")

    # Apply art style
    styled_image = df.apply_art_style(input_image_path, generator)

    # Save or display styled image
    df.save_or_display_image(styled_image, save=True, display=True)

if __name__ == "__main__":
    main()
