#train.py
#code from GAN cookbook CH 6 plus my own edits

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
import numpy as np

class Trainer:
    def __init__(self, height = 64, width = 64, epochs = 50000, batch = 32, checkpoint = 50, train_data_path_A = '',train_data_path_B =   '',test_data_path_A='',test_data_path_B=''):
        #instantiating input variables
        self.EPOCHS = epochs
        self.BATCH = batch
        self.RESIZE_HEIGHT = height
        self.RESIZE_WIDTH = width
        self.CHECKPOINT = checkpoint
        
        #loading data into respective class variables
        self.X_train_A, self.H_A, self.W_A, self.C_A = self.load_data(train_data_path_A)
        self.X_train_B, self.H_B, self.W_B, self.C_B = self.load_data(train_data_path_B)
        self.X_test_A, self.H_A_test, self.W_A_test, self.C_A_test = self.load_data(test_data_path_A)
        self.X_test_B, self.H_B_test, self.W_B_test, self.C_B_test = self.load_data(test_data_path_B)

        #generators
        self.generator_A_to_B = Generator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.generator_B_to_A = Generator(height=self.H_B, width=self.W_B, channels=self.C_B)

        self.orig_A = Input(shape=(self.W_A, self.H_A, self.C_A))
        self.orig_B = Input(shape=(self.W_B, self.H_B, self.C_B))

        self.fake_B = self.generator_A_to_B.Generator(self.orig_A)
        self.fake_A = self.generator_B_to_A.Generator(self.orig_B)
        self.reconstructed_A = self.generator_B_to_A.Generator(self.fake_B)
        self.reconstructed_B = self.generator_A_to_B.Generator(self.fake_A)
        self.id_A = self.generator_B_to_A.Generator(self.orig_A)
        self.id_B = self.generator_A_to_B.Generator(self.orig_B)

        #discriminators
        self.discriminator_A = Discriminator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.discriminator_B = Discriminator(height=self.H_B, width=self.W_B, channels=self.C_B)
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.valid_A = self.discriminator_A.Discriminator(self.fake_A)
        self.valid_B = self.discriminator_B.Discriminator(self.fake_B)

        #passing models onto the GAN
        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid_A, self.valid_B,self.reconstructed_A,self.reconstructed_B,self.id_A, self.id_B]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs,lambda_cycle=10.0,lambda_id=1.0)

    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)
            
            print(f'Starting Epoch {e+1}/{self.EPOCHS}')
            
            while min(len(X_train_A_temp),len(X_train_B_temp))>self.BATCH:
                # Keep track of Batches
                b=b+1

                print(f'Starting Batch {b}')

                # Train Discriminator
                # Grab Real Images for this training batch
                count_real_images = int(self.BATCH)
                starting_indexs = randint(0, 
                (min(len(X_train_A_temp),len(X_train_B_temp))-count_real_images))
                real_images_raw_A = X_train_A_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                real_images_raw_B = X_train_B_temp[ starting_indexs : (starting_indexs + count_real_images) ]

                # Delete the images used until we have none left
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_indexs, (starting_indexs + count_real_images)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_indexs, (starting_indexs + count_real_images)),0)
                batch_A = real_images_raw_A.reshape( count_real_images, self.W_A, self.H_A, self.C_A )
                batch_B = real_images_raw_B.reshape( count_real_images, self.W_B, self.H_B, self.C_B )

                if self.flipCoin():
                    x_batch_A = batch_A
                    x_batch_B = batch_B
                    y_batch_A = np.ones([count_real_images,1])
                    y_batch_B = np.ones([count_real_images,1])
                else:
                    x_batch_B = self.generator_A_to_B.Generator.predict(batch_A)
                    x_batch_A = self.generator_B_to_A.Generator.predict(batch_B)
                    y_batch_A = np.zeros([self.BATCH,1])
                    y_batch_B = np.zeros([self.BATCH,1])
                
                # Now, train the discriminator with this batch
                self.discriminator_A.Discriminator.trainable = True
                discriminator_loss_A = self.discriminator_A.Discriminator.train_on_batch(x_batch_A,y_batch_A)[0]
                self.discriminator_A.Discriminator.trainable = False
                self.discriminator_B.Discriminator.trainable = True
                discriminator_loss_B = self.discriminator_B.Discriminator.train_on_batch(x_batch_B,y_batch_B)[0]          
                self.discriminator_B.Discriminator.trainable = False

                 # Print discriminator loss
                print(f'Batch {b}, Epoch {e+1}: Discriminator A Loss: {discriminator_loss_A}, Discriminator B Loss: {discriminator_loss_B}')

                # In practice, flipping the label when training the generator improves convergence
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels =np.zeros([self.BATCH,1])
                
                generator_loss = self.gan.gan_model.train_on_batch([x_batch_A, x_batch_B],[y_generated_labels, y_generated_labels,x_batch_A, x_batch_B,x_batch_A, x_batch_B])
                # Print generator loss
                print(f'Batch {b}, Epoch {e+1}: Generator Loss: {generator_loss}')

                print ('Batch: '+str(int(b))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)
                    print(f'Checkpoint saved for Batch {b}, Epoch {e+1}')  # Print statement for checkpoint

            print ('Epoch: '+str(int(e))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')

        return

    def load_data(self, data_path, amount_of_data=1.0):
        print(f'Loading data from {data_path}')  # Print statement for starting data loading

        listOFFiles = self.grabListOfFiles(data_path, extension="jpg")
        print(f'Found {len(listOFFiles)} files')  # Print the number of files found

        X_train = np.array(self.grabArrayOfImages(listOFFiles))
        height, width, channels = np.shape(X_train[0])
        print(f'Image shape: Height {height}, Width {width}, Channels {channels}')  # Print image dimensions

        X_train = X_train[:int(amount_of_data * float(len(X_train)))]
        print(f'Using {len(X_train)} images for training (amount_of_data factor: {amount_of_data})')  # Print number of images being used

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        print('Data loading and preprocessing completed')  # Completion statement

        return X_train, height, width, channels


    def grabListOfFiles(self, startingDirectory, extension=".webp"):
        print(f'Searching for files with extension {extension} in {startingDirectory}')  # Starting search

        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))

        print(f'Found {len(listOfFiles)} files')  # Number of files found
        return listOfFiles

    def flipCoin(self,chance=0.5):
        return np.random.binomial(1, chance)

    def grabArrayOfImages(self, listOfFiles, gray=False):
        print(f'Loading and processing {len(listOfFiles)} images')  # Starting image processing

        imageArr = []
        for f in listOfFiles:
            print(f"Attempting to open file: {f}")  # Debug print statement
            try:
                if gray:
                    im = Image.open(f).convert("L")
                else:
                    im = Image.open(f).convert("RGB")
            except FileNotFoundError:
                print(f"File not found: {f}")  # Print statement if file is not found
                continue  # Skip the current file and continue with the next
            im = im.resize((self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
            imData = np.asarray(im)
            imageArr.append(imData)

        print('Image processing completed')  # Completion of image processing
        return imageArr

    def plot_checkpoint(self,b):
        print(f'Creating checkpoint for batch {b}')  # Starting checkpoint creation

        orig_filename = "/data/batch_check_"+str(b)+"_original.png"

        image_A = self.X_test_A[5]
        image_A = np.reshape(image_A, [self.W_A_test,self.H_A_test,self.C_A_test])
        fake_B = self.generator_A_to_B.Generator.predict(image_A.reshape(1, self.W_A, self.H_A, self.C_A ))
        fake_B = np.reshape(fake_B, [self.W_A_test,self.H_A_test,self.C_A_test])
        reconstructed_A = self.generator_B_to_A.Generator.predict(fake_B.reshape(1, self.W_A, self.H_A, self.C_A ))
        reconstructed_A = np.reshape(reconstructed_A, [self.W_A_test,self.H_A_test,self.C_A_test])
        checkpoint_images = np.array([image_A, fake_B, reconstructed_A])

        # Rescale images 0 - 1
        checkpoint_images = 0.5 * checkpoint_images + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axes = plt.subplots(1, 3)
        for i in range(3):
                image = checkpoint_images[i]
                image = np.reshape(image,    
                        [self.H_A_test,self.W_A_test,self.C_A_test])
                axes[i].imshow(image)
                axes[i].set_title(titles[i])
                axes[i].axis('off')
        fig.savefig("/data/batch_check_"+str(b)+".png")
        plt.close('all')
        print(f'Checkpoint image for batch {b} saved')  # Completion of checkpoint creation

        return
