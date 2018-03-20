########################################################################
# Utility functions for converting numpy arrays to RLE and vice versa
#
# Author:   Michael Nguyen
# Email:    mn2769@columbia.edu
# Date:     3/20/18
########################################################################

import numpy as np

########################################################################
# This function converts a numpy array (img) to a run line encoding.
# Logic similar to that of Kaggle user Rakhlin's kernel
# Input: 
# 	- x: 2D numpy array of image with shape (height, width)
#		 1 = mask, 0 = background
# 		 We assume a single image contains only one mask
# Output:
# 	- Returns RLE a space delimited string
########################################################################
def img2rle( x ):
	
	# Find indices of mask locations
	x_flat = x.flatten('F')
	points = np.where(x_flat > 0)[0]

	# Record locations in RLE format
	rle = []
	prev = -2
	for loc in points:
		# Add the new start location if not contiguous
		if loc > prev + 1:
			rle.extend( (loc+1, 0) )

		# Extend run length
		rle[-1] += 1
		prev = loc

	# Print out in space delimited format
	return ' '.join([str(loc) for loc in rle])

########################################################################
# This function converts run line encoding to a numpy array (img).
# Input:
# 	- rle: RLE a space delimited string
#	- shape: list indicating (height, width) of the image
# Output: 
# 	- 2D numpy array of image with shape (height, width)
#		 1 = mask, 0 = background
# 		 We assume a single image contains only one mask
########################################################################
def rle2img( rle, shape ):

	# Map RLE pairs to ints in a list 
	rle_list = list(  map(int, rle.split(' ')) )

	# Use a flattened matrix to record masks
	x = np.zeros( shape[0] * shape[1] )

	# Record mask values in x
	for iter in range( int( len(rle_list)/2 ) ):
		start_ind = rle_list[iter*2]
		end_ind = start_ind + rle_list[iter*2 + 1] - 1

		x[start_ind:(end_ind + 1)] = 1

	# Reshape the matrix to shape
	return np.reshape(x, shape, order='F')
