#/User/yaofeiwang/tensorflow//bin python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import sys
import os
import h5py
from keras.preprocessing import image
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout
from PIL import Image
import scipy.misc
import cv2
import pickle
from keras.optimizers import SGD
import time


#################### import images, represented as Num*size(0)*size(1)*channels

def loadImage(filePath, imageSize = (200, 140)):
    """
    input arguements: 
        filePath - target filefolder where the images are in
        imageSize - read different size of images in to a uniform size
    
    output:
        data - a 4D array contains all the image data (number of images, size[0], size[1], number of channels)
        label - label of corresponding input image (1: no makeup; 2: with makeup)
        imgNum - total number of input images
        
    """
    #calculate the total number of the images inside the target filefoder
    imgNum = 0
    for filename in os.listdir(filePath):
        if  filename != 'Thumbs.db':
            imgNum = imgNum + 1
    
    #generate an original empty 4D array for input images    
    data = np.empty((imgNum,200,140,3), dtype="float32")

    #load images, and extract each label from their file name
    img_count=0
    label = []
    for filename in os.listdir(filePath):
        filename_split = filename.split('.')
        #read only 'jpg' file
        if  filename != 'Thumbs.db':
        #if filename_split[-1] == 'jpg' or 'png':   
            #load image as PIL image mode
            img = image.load_img(filePath+filename, target_size=imageSize)   
            #transform PIL image into array
            x = image.img_to_array(img)
            #add dimision into image to gather data
            x = np.expand_dims(x, axis=0)
            data[img_count] = x
            label.append(int(filename_split[-2]))
            img_count = img_count+1 
    return data, np.array(label), imgNum

#load images
img_path = '/Users/yaofeiwang/Dropbox/asian_label/'#asianLabed\\
data, label, imgNum = loadImage(img_path)



################### build model
start = time.time()
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def baseline_model():
	# create model
	model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(data.shape[1],data.shape[2],data.shape[3]), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Dropout(0.2))
	model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.2))
	model.add(Dense(1, activation='relu'))
	sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        # Compile model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
print(label)
model.fit(data, label, validation_data=(data, label), nb_epoch=10, batch_size=5, verbose=2)
#save the model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# Final evaluation of the model
scores = model.evaluate(data,label, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
end = time.time()
print(end - start)
