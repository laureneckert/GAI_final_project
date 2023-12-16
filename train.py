#train.py
#code from GAN cookbook CH 6 plus my own edits

from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.layers import Input
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, height, width, channels, epochs, batch, checkpoint, train_data_path_A, train_data_path_B):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint
        
        # Store paths as attributes
        self.train_data_path_A = train_data_path_A
        self.train_data_path_B = train_data_path_B
        
        # Define inputs
        self.orig_A = Input(shape=(self.W, self.H, self.C))
        self.orig_B = Input(shape=(self.W, self.H, self.C))
        
        # Initialize batch generators for training data
        self.train_generator_A = self.load_data(train_data_path_A)
        self.train_generator_B = self.load_data(train_data_path_B)

        # Initialize models
        self.generator = Generator(height=self.H, width=self.W, channels=self.C)
        self.fake_A = self.generator.Generator(self.orig_B)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.discriminator.trainable = False
        self.valid = self.discriminator.Discriminator([self.fake_A, self.orig_B])

        model_inputs = [self.orig_A, self.orig_B]
        model_outputs = [self.valid, self.fake_A]
        self.gan = GAN(model_inputs=model_inputs, model_outputs=model_outputs)

    def load_data(self, data_path):
        return self.create_data_generator(data_path)

    def create_data_generator(self, data_path):
        list_of_files = self.grabListOfFiles(data_path, extension="jpg")
        def image_batch_generator():
            i = 0
            while True:
                batch_files = list_of_files[i:i + self.BATCH]
                if len(batch_files) < self.BATCH:
                    # Not enough images to form a batch, reset index to start
                    i = 0
                    continue
                batch_images = []
                for file_path in batch_files:
                    try:
                        image = Image.open(file_path).convert('RGB')
                        image = image.resize((self.W, self.H))
                        image_array = np.asarray(image, dtype=np.float32)
                        image_array = (image_array - 127.5) / 127.5
                        batch_images.append(image_array)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                yield np.array(batch_images)
                i = (i + self.BATCH) % len(list_of_files)  # Move index, reset if end of list reached

        return image_batch_generator

    def grabListOfFiles(self, startingDirectory, extension=".jpg"):
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                fullPath = os.path.join(startingDirectory, file)
                print("Loading file:", fullPath)  # Debug print
                listOfFiles.append(fullPath)
        return listOfFiles

    def grabArrayOfImages(self, listOfFiles, gray=False):
        imageArr = []
        for f in listOfFiles:
            try:
                print(f"Attempting to open image: {f}")
                if gray:
                    im = Image.open(f).convert("L")
                else:
                    im = Image.open(f).convert("RGB")
                imData = np.asarray(im)
                imageArr.append(imData)
                print(f"Loaded image: {f}")
            except IOError as e:
                print(f"Error opening file {f}: {e}")
            except Exception as e:
                print(f"Unexpected error with file {f}: {e}")
        return imageArr

    def train(self):
        print("Starting training process")

        for epoch in range(self.EPOCHS):
            print(f"Epoch {epoch+1}/{self.EPOCHS}")

            # Initialize generators
            self.gen_A = self.train_generator_A()
            self.gen_B = self.train_generator_B()

            batch_index = 0
            while True:
                try:
                    batch_A = next(self.gen_A)
                    batch_B = next(self.gen_B)
                except StopIteration:
                    # Reset the generators if exhausted and break the loop
                    self.gen_A = self.train_generator_A()
                    self.gen_B = self.train_generator_B()
                    break  # Break the loop if any of the generators is exhausted

                print(f"\nProcessing batch {batch_index + 1} of epoch {epoch + 1}")
                
                # Debugging: Check shapes and sample values
                print(f"Shape of batch_A: {batch_A.shape}")
                print(f"Sample value from batch_A: {batch_A[0,0,0,:]}")  # Print a small sample
                print(f"Shape of batch_B: {batch_B.shape}")
                print(f"Sample value from batch_B: {batch_B[0,0,0,:]}")  # Print a small sample

                # PatchGAN target labels for real and fake images
                y_valid = np.ones((batch_A.shape[0],) + (int(self.W / 2**4), int(self.W / 2**4), 1))
                y_fake = np.zeros((batch_B.shape[0],) + (int(self.W / 2**4), int(self.W / 2**4), 1))
                print("Log 1")

                # Generate a batch of new images (fake images)
                fake_A = self.generator.Generator.predict(batch_B)
                print("Log 2")

                # Debugging: Check fake_A
                print(f"Shape of fake_A: {fake_A.shape}")
                print(f"Sample value from fake_A: {fake_A[0,0,0,:]}")  # Print a small sample

                # Train the discriminator (real classified as ones and generated as zeros)
                print("Training discriminator with real data...")
                try:
                    discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A, batch_B], y_valid)[0]
                    print("Log 3 - Discriminator real")
                except Exception as e:
                    print(f"Error during discriminator training with real data: {e}")
                
                # Train discriminator with fake data
                print("Training discriminator with fake data...")
                try:
                    discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A, batch_B], y_fake)[0]
                    print("Log 4 - Discriminator fake")
                except Exception as e:
                    print(f"Error during discriminator training with fake data: {e}")
                full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)


                # Train the generator
                generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B], [y_valid, batch_A])
                print("Log 5 - Generator")

                print(f'Batch {batch_index+1}: [Discriminator Loss: {full_loss}], [Generator Loss: {generator_loss}]')

                if batch_index % self.CHECKPOINT == 0:
                    print("Checkpoint reached. Checkpoint saving skipped.")
                    #label = f"{epoch}_{batch_index}"
                    #print(f"Checkpoint reached: Saving models and generating plot for epoch {epoch + 1}, batch {batch_index + 1}")
                    #self.plot_checkpoint(label)

                batch_index += 1  # Increment batch index
                
                # Optionally reset the generators at the end of each epoch
                self.gen_A = self.train_generator_A()
                self.gen_B = self.train_generator_B()
            print(f'Epoch {epoch+1} completed: [Discriminator Loss: {full_loss}], [Generator Loss: {generator_loss}]')

        print("Training completed")

    def plot_checkpoint(self, b):
            # Ensuring the 'out' directory exists
            output_dir = 'out'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Fetching one batch of images from each generator
            try:
                imgs_A = next(self.gen_A)
                imgs_B = next(self.gen_B)
            except StopIteration:
                # Handle the case if the generator is exhausted
                print("Generators exhausted. Unable to plot images.")
                return

            # Generate fake images
            fake_A = self.generator.Generator.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            r, c = 3, 3
            titles = ['Style', 'Generated', 'Original']  # Define titles here
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    if cnt < len(gen_imgs):  # Check to avoid going out of bounds
                        axs[i, j].imshow(gen_imgs[cnt])
                        axs[i, j].set_title(titles[i % len(titles)])  # To cycle through titles
                        axs[i, j].axis('off')
                        cnt += 1
                    else:
                        break  # Break the inner loop if we've processed all images

            # Save the figure
            output_file = os.path.join(output_dir, f"batch_check_{b}.png")
            fig.savefig(output_file)
            plt.close('all')

            return output_file
