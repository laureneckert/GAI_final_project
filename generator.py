#generator.py
#code from GAN cookbook CH 6 ebook 

import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate, LayerNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras.utils import plot_model

class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1):

        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (width,height,channels)

        self.Generator = self.model()
        self.OPTIMIZER = Adam(lr=1e-4, beta_1=0.2)
        self.Generator.compile(loss='binary_crossentropy', 
               optimizer=self.OPTIMIZER,metrics=['accuracy'])

        self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(shape=self.SHAPE)

        down_1 = Convolution2D(32  , kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
        norm_1 = LayerNormalization()(down_1) #instance normalization

        down_2 = Convolution2D(32*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_1)
        norm_2 = LayerNormalization()(down_2)

        down_3 = Convolution2D(32*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 = LayerNormalization()(down_3)

        down_4 = Convolution2D(32*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 = LayerNormalization()(down_4)

        upsample_1 = UpSampling2D()(norm_4)
        up_conv_1 = Convolution2D(32*4, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_1)
        norm_up_1 = LayerNormalization()(up_conv_1)
        add_skip_1 = Concatenate()([norm_up_1,norm_3])

        upsample_2 = UpSampling2D()(add_skip_1)
        up_conv_2 = Convolution2D(32*2, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_2)
        norm_up_2 = LayerNormalization()(up_conv_2)
        add_skip_2 = Concatenate()([norm_up_2,norm_2])

        upsample_3 = UpSampling2D()(add_skip_2)
        up_conv_3 = Convolution2D(32, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_3)
        norm_up_3 = LayerNormalization()(up_conv_3)
        add_skip_3 = Concatenate()([norm_up_3,norm_1])

        last_upsample = UpSampling2D()(add_skip_3)
        output_layer = Convolution2D(3, kernel_size=4, strides=1, 
        padding='same',activation='tanh')(last_upsample)

        return Model(input_layer,output_layer)
    
    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        try:
            plot_model(self.Generator, to_file=r'C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\GAI_final_project\out\Generator_Model.png')
        except Exception as e:
            print(f"An error occurred while saving the model diagram: {e}")