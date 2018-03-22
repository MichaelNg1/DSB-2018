########################################################################
# Load in the model and apply to the test data
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/20/18
########################################################################

import sys
import numpy as np

from keras.models import Model, load_model

import tensorflow as tf

sys.path.insert(0, '../util')
import convert as con
import unet_tools as tool
from unet_tools import mean_iou

########################################################################
# Global Constants
########################################################################
MODEL_PATH = 'model-dsbowl2018-1.h5'
TEST_PATH = '../../data/stage1_test/'
TRAIN_PATH = '../../data/stage1_train/'
TRUTH_PATH = '../../data/stage1_train/stage1_train_labels.csv'
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

if __name__ == '__main__':

	x_test = tool.process_testing(TEST_PATH, 
		IMG_HEIGHT, 
		IMG_WIDTH, 
		IMG_CHANNELS)

	x_train, y_train = tool.process_training(TRAIN_PATH, 
		TRUTH_PATH, 
		IMG_HEIGHT, 
		IMG_WIDTH, 
		IMG_CHANNELS)

	# Predict on train, val and test
	model = load_model(MODEL_PATH, custom_objects={'mean_iou': mean_iou})
	preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
	preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
	preds_test = model.predict(x_test, verbose=1)

	# Threshold predictions
	preds_train_t = (preds_train > 0.5).astype(np.uint8)
	preds_val_t = (preds_val > 0.5).astype(np.uint8)
	preds_test_t = (preds_test > 0.5).astype(np.uint8)

	# Create list of upsampled test masks
	# preds_test_upsampled = []
	# for i in range(len(preds_test)):
	#     preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
	#                                        (sizes_test[i][0], sizes_test[i][1]), 
	#                                        mode='constant', preserve_range=True))