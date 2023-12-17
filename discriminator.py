#discriminator.py
#code from GAN cookbook CH 6 ebook

import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, concatenate, LayerNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,Nadam, Adamax
import keras.backend as K
from keras.utils import plot_model


class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)

        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(self.SHAPE)
        
        up_layer_1 = Convolution2D(64, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

        up_layer_2 = Convolution2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        norm_layer_1 = LayerNormalization()(up_layer_2)

        up_layer_3 = Convolution2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_1)
        norm_layer_2 = LayerNormalization()(up_layer_3)

        up_layer_4 = Convolution2D(64*8, kernel_size=4, strides=2, 
        padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_2)
        norm_layer_3 = LayerNormalization()(up_layer_4)

        output_layer = Convolution2D(1, kernel_size=4, strides=1, padding='same')(norm_layer_3)
        output_layer_1 = Flatten()(output_layer)
        output_layer_2 = Dense(1, activation='sigmoid')(output_layer_1)

        return Model(input_layer,output_layer_2)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        try:
            plot_model(self.Discriminator, to_file=r'C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\GAI_final_project\out\Discriminator_Model.png')
        except Exception as e:
            print(f"An error occurred while saving the model diagram: {e}")

