########################################################################
# Visualize a single specified image w/ and w/o truth mask
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/20/18
########################################################################

import pathlib
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import cv2 as cv

# import custom scripts
import util.convert as conv
import util.unet_tools as tool

# Fixed variables
TEST_PATH = '../data/stage1_test/'
TRAIN_PATH = '../data/stage1_train/'
TRUTH_PATH = '../data/stage1_train/stage1_train_labels.csv'
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
NUM_VISUAL = 5

x_test = tool.process_testing(TEST_PATH, 
	IMG_HEIGHT, 
	IMG_WIDTH, 
	IMG_CHANNELS)

x_train, y_train = tool.process_training(TRAIN_PATH, 
	TRUTH_PATH, 
	True,
	IMG_HEIGHT, 
	IMG_WIDTH, 
	IMG_CHANNELS)

x_train_old, y_train_old = tool.process_training(TRAIN_PATH, 
	TRUTH_PATH, 
	False,
	IMG_HEIGHT, 
	IMG_WIDTH, 
	IMG_CHANNELS)

# Visualize random images from training img, training mask, test img
fig, ax = plt.subplots(NUM_VISUAL,3)
ind_train = np.random.randint(0, high=x_train.shape[0], size=NUM_VISUAL)

for k in range( int(NUM_VISUAL) ):
	ind = random.randint(0, x_train.shape[0])
	if k == 0:
		ax[k][0].set_title('Original mask')
	ax[k][0].imshow( np.squeeze(y_train_old[ ind_train[k] ] ) )
	

	if k == 0:
		ax[k][1].set_title('Eroded mask')
	ax[k][1].imshow( np.squeeze(y_train[ ind_train[k] ]))

	if k == 0:
		ax[k][2].set_title('Original w/ eroded mask')
	ax[k][2].imshow( x_train[ ind_train[k] ] )
	ax[k][2].imshow( np.squeeze(y_train[ ind_train[k] ]), alpha = 0.25)

plt.show()