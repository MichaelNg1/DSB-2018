########################################################################
# Utility functions for UNet implementation
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/20/18
########################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

from skimage.io import imread
from skimage.transform import resize

from keras import backend as K
import tensorflow as tf

import convert as con

# Folder for all UNet pre-processed data
PP_PATH = '../../data/preprocessed_data/'

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

########################################################################
# This function converts training images and their masks to downsampled
# images which are converted to numpy arrays. The images and masks are
# returned and saved in the preprocessed data folder
# Input: 
# 	- PATH_IMG: path to the training images
#	- PATH_RLE: path to the training mask csv
#	- IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS: image downsampling parameters
# Output:
# 	- x_train, y_train: numpy arrays of the downsampled images/masks
########################################################################
def process_training(PATH_IMG, PATH_RLE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

	# Create the filename for training data
	header = list( filter(None, PATH_IMG.split('/')) )[-1]
	x_name = '_'.join([header, 
		str(IMG_HEIGHT), 
		str(IMG_WIDTH), 
		str(IMG_CHANNELS), 
		'train_img'])
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
		training_paths = pathlib.Path( PATH_IMG ).glob('*/images/*.png')
		training_paths = sorted([str(x) for x in training_paths])
		training_id = [ x.split('/')[-3] for x in training_paths]
		training_len = len( training_id )

		# Set up the empty arrays for training images (x) and label/masks (y)
		x_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
		y_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

		# Get the data from the truth and select the masks w/ relevant data
		df_mask = pd.read_csv(PATH_RLE)

		# Get and resize the training images
		print('Processing Images:')
		for ind_, path_ in tqdm( enumerate(training_paths), total=training_len ):
			
			# record the downsized image
			img = imread(path_)[:, :, :IMG_CHANNELS]
			orig_mask_shape = img.shape[0:2]
			img = resize(img, (IMG_HEIGHT, IMG_WIDTH),  mode='constant', preserve_range=True)
			x_train[ind_] = img

			# collect all mask information by taking the union over all masks
			df_filt = df_mask.loc[ df_mask['ImageId'] == training_id[ind_] ]
			mask = np.zeros(orig_mask_shape , dtype=np.bool)
			for row in df_filt.iterrows():
				rle = str(row[1]['EncodedPixels'])
				np_mask = con.rle2img( rle, orig_mask_shape )
				mask = np.maximum(mask, np_mask)
			mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1),  mode='constant', preserve_range=True)	
			y_train[ind_] = mask

		# Save the numpy, downsized training images and masks
		np.save(PP_PATH + x_name, x_train)
		np.save(PP_PATH + y_name, y_train)
	return x_train, y_train

########################################################################
# This function converts testing images to downsampled
# images which are converted to numpy arrays. The images are
# returned and saved in the preprocessed data folder
# Input: 
# 	- PATH_IMG: path to the testing images
#	- IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS: image downsampling parameters
# Output:
# 	- x_test: numpy arrays of the downsampled images
########################################################################
def process_testing(PATH_IMG, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
	# Create the filename for training data
	header = list( filter(None, PATH_IMG.split('/')) )[-1]
	x_name = '_'.join([header, 
		str(IMG_HEIGHT), 
		str(IMG_WIDTH), 
		str(IMG_CHANNELS), 
		'test_img'])

	# Check to see if we pre-processed the data with the parameters before
	if pathlib.Path(PP_PATH + x_name + '.npy').exists():
		print('Preprocessed data found')
		x_test = np.load(PP_PATH + x_name + '.npy')
	else:
		# Grab the paths and ids of all training images
		testing_paths = pathlib.Path( PATH_IMG ).glob('*/images/*.png')
		testing_paths = sorted([str(x) for x in testing_paths])
		testing_id = [ x.split('/')[-3] for x in testing_paths]
		testing_len = len( testing_id )

		# Set up the empty arrays for training images (x) and label/masks (y)
		x_test = np.zeros((testing_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

		# Get and resize the training images
		print('Processing Images:')
		for ind_, path_ in tqdm( enumerate(testing_paths), total=testing_len ):
			
			# record the downsized image
			img = imread(path_)[:, :, :IMG_CHANNELS]
			img = resize(img, (IMG_HEIGHT, IMG_WIDTH),  mode='constant', preserve_range=True)
			x_test[ind_] = img

		# Save the numpy, downsized training images and masks
		np.save(PP_PATH + x_name, x_test)
	return x_test
