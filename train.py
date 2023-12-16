#train.py
#code from GAN cookbook CH 6

from gan import GAN
from generator import Generator
from discriminator import Discriminator
from keras.layers import Input
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from PIL import Image
import random
import numpy as np


class Trainer:
    def __init__(self, height, width, channels, epochs, batch, checkpoint,
                 train_data_path_A, train_data_path_B, test_data_path_A, test_data_path_B):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint

        # Load data for both domains
        self.train_generator_A = self.load_data(train_data_path_A)
        self.train_generator_B = self.load_data(train_data_path_B)
        self.test_generator_A = self.load_data(test_data_path_A)
        self.test_generator_B = self.load_data(test_data_path_B)

        # Initialize the rest of your models here
        self.generator = Generator(height=self.H, width=self.W, channels=self.C)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)

        self.discriminator.trainable = False
        self.valid = self.discriminator.Discriminator([self.fake_A,self.orig_B])

        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid, self.fake_A]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs)
        
    def load_data(self, data_path):
        list_of_files = self.grabListOfFiles(data_path, extension="jpg")
        batch_size = 100  # Adjust this based on your system's capability

        def image_batch_generator():
            for i in range(0, len(list_of_files), batch_size):
                batch_files = list_of_files[i:i + batch_size]
                batch_images = []
                for file_path in batch_files:
                    try:
                        image = Image.open(file_path).convert('RGB')
                        image = image.resize((self.W, self.H))  # Resize image
                        image_array = np.asarray(image, dtype=np.float32)
                        image_array = (image_array - 127.5) / 127.5  # Normalize image
                        batch_images.append(image_array)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                yield np.array(batch_images)  # Yields a batch of normalized images

        return image_batch_generator()

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

            # Initialize batch generators
            self.train_generator_A = self.load_data(self.train_data_path_A)
            self.train_generator_B = self.load_data(self.train_data_path_B)

            for epoch in range(self.EPOCHS):
                print(f"Epoch {epoch+1}/{self.EPOCHS}")

                # Iterate over the batch generators
                for batch_index, (batch_A, batch_B) in enumerate(zip(self.train_generator_A, self.train_generator_B)):
                    print(f"\nProcessing batch {batch_index + 1} of epoch {epoch + 1}")

                    # PatchGAN target labels for real and fake images
                    y_valid = np.ones((batch_A.shape[0],) + (int(self.W / 2**4), int(self.W / 2**4), 1))
                    y_fake = np.zeros((batch_B.shape[0],) + (int(self.W / 2**4), int(self.W / 2**4), 1))

                    # Generate a batch of new images (fake images)
                    fake_A = self.generator.Generator.predict(batch_B)

                    # Train the discriminator (real classified as ones and generated as zeros)
                    discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A, batch_B], y_valid)[0]
                    discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A, batch_B], y_fake)[0]
                    full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

                    # Train the generator
                    generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B], [y_valid, batch_A])

                    print(f'Batch {batch_index+1}: [Discriminator Loss: {full_loss}], [Generator Loss: {generator_loss}]')

                    if batch_index % self.CHECKPOINT == 0:
                        label = f"{epoch}_{batch_index}"
                        print(f"Checkpoint reached: Saving models and generating plot for epoch {epoch + 1}, batch {batch_index + 1}")
                        self.plot_checkpoint(label)

                print(f'Epoch {epoch+1} completed: [Discriminator Loss: {full_loss}], [Generator Loss: {generator_loss}]')

            print("Training completed")

    def plot_checkpoint(self,b):
        orig_filename = "/out/batch_check_"+str(b)+"_original.png"

        r, c = 3, 3
        random_inds = random.sample(range(len(self.X_test_A)),3)
        imgs_A = self.X_test_A[random_inds].reshape(3, self.W, self.H, self.C )
        imgs_B = self.X_test_B[random_inds].reshape( 3, self.W, self.H, self.C )
        fake_A = self.generator.Generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Style', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/out/batch_check_"+str(b)+".png")
        plt.close('all')

        return