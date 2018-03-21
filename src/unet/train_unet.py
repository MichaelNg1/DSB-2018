########################################################################
# U-Net adopted from the Kaggle kernel by Kjetil Amdal-Saevik
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
TRUTH_PATH = '../../data/stage1_train_labels.csv'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

########################################################################
# Process the data:
#	Here we extract the images from the files and create a boolean mask
#	from the csv file. The images are then downsized.
#
########################################################################

# Grab the paths and ids of all training images
training_paths = pathlib.Path( TRAIN_PATH ).glob('*/images/*.png')
training_paths = sorted([str(x) for x in training_paths])
training_id = [ x.split('/')[-3] for x in training_paths]
training_len = len( training_id )

# Get the data from the truth and select the masks w/ relevant data
df_mask = pd.read_csv(TRUTH_PATH)

# Set up the training images (x) and label/masks (y)
x_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_train = np.zeros((training_len, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

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

# Visualize a random mask
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title('Raw Image')
ax1.imshow( x_train[1], cmap='gray' )

ax2.set_title('Image with Truth Mask')
ax2.imshow( x_train[1], cmap='gray' )
ax2.imshow( np.squeeze(y_train[1]), alpha=0.5)
plt.show()