########################################################################
# U-Net implementation
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/21/18
########################################################################

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
########################################################################
# Global Constants
########################################################################
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../../data/stage1_train/'
TEST_PATH = '../../data/stage1_test/'
TRUTH_PATH = '../../data/stage1_train/stage1_train_labels.csv'
PP_PATH = '../../data/preprocessed_data/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

if __name__ == '__main__':

	########################################################################
	# Process the data:
	#	Here we extract the images from the files and create the mask info
	#	from the csv file. The images are downsized and converted to
	#	numpy arrays.
	########################################################################

	# Create the filename for training data
	header = list( filter(None, TRAIN_PATH.split('/')) )[-1]
	x_name = '_'.join([header, 
		str(IMG_HEIGHT), 
		str(IMG_WIDTH), 
		str(IMG_CHANNELS), 
		'images'])
	y_name = '_'.join([header, 
		str(IMG_HEIGHT), 
		str(IMG_WIDTH), 
		str(IMG_CHANNELS), 
		'masks'])

	# Check to see if we pre-processed the data with the parameters before
	if pathlib.Path(PP_PATH + x_name + '.npy').exists() \
		and pathlib.Path(PP_PATH + y_name + '.npy').exists():
		print('Preprocessed data found')
		x_train = np.load(PP_PATH + x_name + '.npy')
		y_train = np.load(PP_PATH + y_name + '.npy')
	else:
		# Grab the paths and ids of all training images
		training_paths = pathlib.Path( TRAIN_PATH ).glob('*/images/*.png')
		training_paths = sorted([str(x) for x in training_paths])
		training_id = [ x.split('/')[-3] for x in training_paths]
		training_len = len( training_id )

		# Set up the empty arrays for training images (x) and label/masks (y)
		x_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
		y_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

		# Get the data from the truth and select the masks w/ relevant data
		df_mask = pd.read_csv(TRUTH_PATH)

		# Get and resize the training images
		print('Processing Images:')
		for ind, path_ in tqdm( enumerate(training_paths), total=training_len ):
			
			# record the downsized image
			img = imread(path_)[:, :, :IMG_CHANNELS]
			orig_mask_shape = img.shape[0:2]
			img = resize(img, (IMG_HEIGHT, IMG_WIDTH),  mode='constant', preserve_range=True)
			x_train[ind] = img

			# collect all mask information by taking the union over all masks
			df_filt = df_mask.loc[ df_mask['ImageId'] == training_id[ind] ]
			mask = np.zeros(orig_mask_shape , dtype=np.bool)
			for row in df_filt.iterrows():
				rle = str(row[1]['EncodedPixels'])
				np_mask = con.rle2img( rle, orig_mask_shape )
				mask = np.maximum(mask, np_mask)
			mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1),  mode='constant', preserve_range=True)	
			y_train[ind] = mask

		# Save the numpy, downsized training images and masks
		np.save(PP_PATH + x_name, x_train)
		np.save(PP_PATH + y_name, y_train)

	# Build U-Net model adopted from keegil kaggle kernel
	inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	s = Lambda(lambda x: x / 255) (inputs)

	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.1) (c8)
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

	# Fit model
	earlystopper = EarlyStopping(patience=5, verbose=1)
	checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
	results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=50, 
	                    callbacks=[earlystopper, checkpointer])