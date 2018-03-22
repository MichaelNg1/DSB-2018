# DSB-2018

This repository includes all code for the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). All algorithms/code is still underdevelopment.

## UNet
This contains the model and code to run/train a UNet to generate a Nuclei mask given an aribitrary scan (based on [Saevik's Kaggle Kernel](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277)).

## Utils
This folder contains essential functions relevant for scoring performance, converting to run line encoding (RLE) format, etc.
* **convert**: This script contains functions that convert RLE strings to 2D numpy arrays and vice-versa.
* **unet_tools**: This script contains the functions to preprocess the data (i.e. convert image files or csv files in RLE format to numpy and downsample) and the custom intersection over union (IoU) metric for Keras. The truth masks can be 
