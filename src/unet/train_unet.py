########################################################################
# U-Net implementation
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/21/18
########################################################################

import os
import pathlib
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

sys.path.insert(0, '../util')
import convert as con
import unet_tools as tool
from unet_tools import mean_iou

########################################################################
# Global Constants
########################################################################
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../../data/stage1_train/'
TRUTH_PATH = '../../data/stage1_train/stage1_train_labels.csv'
PP_PATH = '../../data/preprocessed_data/'

seed = 42
random.seed = seed
np.random.seed = seed

if __name__ == '__main__':

	########################################################################
	# Process the data:
	#	Here we extract the images from the files and create the mask info
	#	from the csv file. The images are downsized and converted to
	#	numpy arrays.
	########################################################################

	warnings.simplefilter(action='ignore', category=FutureWarning)
	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

	# Load in the processed training data
	x_train, y_train = tool.process_training(TRAIN_PATH, 
		TRUTH_PATH, 
		IMG_HEIGHT, 
		IMG_WIDTH, 
		IMG_CHANNELS)

	ind = random.randint(0, x_train.shape[0])
	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.set_title('Raw Image')
	ax1.imshow( x_train[ind], cmap='gray' )

	ax2.set_title('Image with Truth Mask')
	ax2.imshow( x_train[ind], cmap='gray' )
	ax2.imshow( np.squeeze( y_train[ind] ), alpha=0.5)
	plt.show()


	# # Build U-Net model adopted from keegil kaggle kernel
	# inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	# s = Lambda(lambda x: x / 255) (inputs)

	# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
	# c1 = Dropout(0.1) (c1)
	# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	# p1 = MaxPooling2D((2, 2)) (c1)

	# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	# c2 = Dropout(0.1) (c2)
	# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	# p2 = MaxPooling2D((2, 2)) (c2)

	# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	# c3 = Dropout(0.2) (c3)
	# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	# p3 = MaxPooling2D((2, 2)) (c3)

	# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	# c4 = Dropout(0.2) (c4)
	# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	# c5 = Dropout(0.3) (c5)
	# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	# u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	# u6 = concatenate([u6, c4])
	# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	# c6 = Dropout(0.2) (c6)
	# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	# u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	# u7 = concatenate([u7, c3])
	# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	# c7 = Dropout(0.2) (c7)
	# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	# u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	# u8 = concatenate([u8, c2])
	# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	# c8 = Dropout(0.1) (c8)
	# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	# u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	# u9 = concatenate([u9, c1], axis=3)
	# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	# c9 = Dropout(0.1) (c9)
	# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	# outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	# model = Model(inputs=[inputs], outputs=[outputs])
	# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

	# # Fit model
	# earlystopper = EarlyStopping(patience=5, verbose=1)
	# checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
	# results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=50, 
	#                     callbacks=[earlystopper, checkpointer])