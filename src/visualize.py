########################################################################
# Visualize a single specified image w/ and w/o truth mask
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/20/18
########################################################################

import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
import matplotlib.pyplot as plt

# import custom scripts
import util.convert as conv

# Fixed variables
DATA_PATH = '../data/stage1_train'
TRUTH_PATH = '../data/stage1_train_labels.csv'

# Change this number for a different image
IMG_INDEX = 125

########################################################################
# 1) Get list of image files and grab one image
########################################################################

# Glob the training data and load a single image path
training_paths = pathlib.Path( DATA_PATH ).glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])

im_path = training_sorted[IMG_INDEX]
im = imageio.imread(str(im_path))

# Grab the image ID
im_id = str(im_path).split('/')[-3]
print('ImageID: %s' % (im_id) )

# Coerce the image into grayscale format (if not already)
im_gray = rgb2gray(im)

########################################################################
# 2) Get mask from the specific image
########################################################################
# Turns out the truth is the same as the png images...

# # Glob the training data and load a single image path
# training_mask_paths = pathlib.Path( DATA_PATH ).glob( im_id + '/masks/*.png')

# mask_cell_tot = np.zeros( im_gray.shape )
# for path in training_mask_paths:
# 	im_mask = np.array( imageio.imread(str(path)) )
# 	mask_cell_tot += rgb2gray(im_mask)

########################################################################
# 3) Get nucleus mask from the specific image
########################################################################

# Get the data from the truth and select the masks w/ relevant data
df = pd.read_csv(TRUTH_PATH)
df_filt = df.loc[df['ImageId'] == im_id]

# Take the union over all the masks, they should be non-overlapping
mask_nucl_tot = np.zeros( im_gray.shape )
for row in df_filt.iterrows():
	rle = str(row[1]['EncodedPixels'])
	np_mask = conv.rle2img( rle, im_gray.shape )
	mask_nucl_tot = np.maximum(mask_nucl_tot, np_mask)

########################################################################
# 4) Visualize results
########################################################################

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title('Raw Image')
ax1.imshow( im_gray, cmap='gray' )

ax2.set_title('Image with Truth Mask')
ax2.imshow( im_gray, cmap='gray' )
ax2.imshow( mask_nucl_tot, alpha=0.5)
plt.show()