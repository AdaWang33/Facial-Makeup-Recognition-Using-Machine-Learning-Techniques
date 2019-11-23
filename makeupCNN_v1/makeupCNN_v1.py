#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Process:
	Load images, split into train and test dataset
	Construct neural networks
	Compile and train with train dataset
	Train with data augmentation
	Test on test dataset
Methods to prevent overfitting:
	In neural networks: pooling, dropout, regularization
	Data augmentation
'''

from keras.preprocessing import image
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils import np_utils
from keras.preprocessing.text import text_to_word_sequence as text2word
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras import metrics

from timeit import default_timer as timer

# Image size
width, height = 200, 140

# Load images
def loadImage(filePath):
    """
    input arguements: 
        filePath - target filefolder where the images are in
        imageSize - read different size of images in to a uniform size
    
    output:
        data - a 4D array contains all the image data (number of images, size[0], size[1], number of channels)
        label - label of corresponding input image (1: no makeup; 2: with makeup)   
    """
	
    #calculate the total number of the images inside the target filefoder
    imgNum = 0
    for filename in os.listdir(filePath):
        if  filename != 'Thumbs.db':
            imgNum = imgNum + 1
    
    #generate an original empty 4D array for input images    
    data = np.empty((imgNum,width,height,3), dtype="float32")

    #load images, and extract each label from their file name
    img_count=0
    label = []
    for filename in os.listdir(filePath):
        filename_split = filename.split('.')  
        #load image as PIL image mode
        img = image.load_img(filePath+filename, target_size=(width,height))   
        #transform PIL image into array
        x = image.img_to_array(img)
        #add dimision into image to gather data
        x = np.expand_dims(x, axis=0)
        data[img_count] = x
        # Labels: images with file name ending with "1" are classified as class 0
		# 	otherwise class 1
        c = text2word(filename_split[-2])
        if c[0] == '1':
            label.append(0)
        else:
            label.append(1)
        img_count = img_count+1 
    return data, np.array(label)

img_path = '/home/groupg/msun/Makeup_Img/' # All images
#train_img_path = '/home/groupg/msun/train/'
#test_img_path = '/home/groupg/msun/test/'

# X: images, y: labels
X_all, y_all = loadImage(img_path)
# For categorical_corssentropy, there are two classes, i.e., 0 and 1
y_all = np_utils.to_categorical(y_all, 2)

# Split data into train dataset (80%) and test dataset (20%)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)
# Another way is to split train data and test data manually
# 	and import from corresponding paths

# fix random seed for reproducibility
np.random.seed(7)

# Create model (neural networks)
# Reference to https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def createModel():
	model = Sequential()
	
	'''
	Convolutional layers
	Inputs are width*height (200*140) RGB (3) images
	ZeroPadding: Add rows and columns of zeros at the top, bottom, left and right side of an image tensor
		Prevent reduction in volume size after CONV
		Improve performance by keeping information at the borders
	Convolution: 64 filters of size 3*3
	BatchNormalization: Normalize the activations of the previous layer at each batch
		Reduce internal covariate shift
	Pooling:
		Reduce amount of parameters or weights
		Dontrol overfitting
	'''
	
	# Block 1
	model.add(ZeroPadding2D((1,1),input_shape=(width,height,3)))
	model.add(Convolution2D(64, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	# Block 2
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	# Block 3
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	# Block 4
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	# Block 5
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3,3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	model.add(Flatten())
	
	'''
	Fully-connected layers
	Dropout: Randomly set a fraction rate of input units to 0 at each update during training
		Prevent overfitting
	Regularizers: Apply penalties on layer parameters or layer activity during optimization
		kernel_regularizer on weights (reduce overfitting), activity_regularizer on biases
	'''
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	
	model.add(Dense(2, activation='softmax',
					kernel_regularizer=regularizers.l2(0.05),
					activity_regularizer=regularizers.l1(0.05),
					))
	return model

# Data augmentation to further enhance performance by preventing overfitting
# Reference to https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Reference to https://keras.io/preprocessing/image/
'''
ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation
	rotation: Degree range for random rotations
	width shift / height shift: Range for random horizontal / vertical shifts
	Shear angle in counter-clockwise direction as radians
		From wikipedia: A shear mapping is a linear map that displaces each point in fixed direction, 
		by an amount proportional to its signed distance from a line that is parallel to that direction
	zoom: Range for random zoom: [lower, upper] = [1-zoom_range, 1+zoom_range]
	horizontal flip: Randomly flip inputs horizontally
	fill mode: Points outside the boundaries of the input are filled
'''
def fitGenModel(model):
	datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
		)
	
	# compute quantities required for featurewise normalization
	datagen.fit(X_train)
	
	# fits the model on batches with real-time data augmentation
	hist1 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
							steps_per_epoch=len(X_train) / 32, epochs=10,
							#validation_data = datagen.flow(X_test, y_test, batch_size=32),
							#validation_steps=len(X_test) / 32,
							validation_data = (X_test, y_test),
							verbose=2,
							)
	return model

def testModel():
	start = timer()
	
	model = createModel()
	
	# Choose adam as optimizer, set the learning rate to 0.0001
	# 	Other optimizers with good performance include RMSprop, Adadelta, Adamax, Nadam
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	# Fit model
	# 	epochs: Number of epochs to train the model
	# 	batch_size: Number of samples per gradient update
	hist = model.fit(X_train, y_train, epochs=30, batch_size=16,
				validation_data = (X_test, y_test),
				verbose=2,
				)
	
	# Write results to file after fit and evaluate
	f = open(filename, 'w')
	f.write("\nfit:\n")
	printResults(model, f)
	
	model = fitGenModel(model)
	
	# Write results to file after fit_generator and evaluate
	f.write("\nfit_generator:\n")
	printResults(model, f)
	
	end = timer()
	
	f.write("\ntime: %d sec\n" % (end - start))
	f.write("\n====================\n")
	
	f.close()
	
	# Save model & weights; HDF5, pip install h5py
	model.save('makeupCNN.h5') 
	model.save_weights('makeupCNN_weights.h5')
	
	return model

# Write evaluate results to file
def printResults(model, f):
	# Evaluate model
	score = model.evaluate(X_train, y_train, verbose=0)
	f.write("Evaluate -- train dataset\nloss: %.2f\tacc: %.2f%%\n" % (score[0], score[1]*100))
	score = model.evaluate(X_test, y_test, verbose=0)
	f.write("Evaluate -- test dataset\nloss: %.2f\tacc: %.2f%%\n" % (score[0], score[1]*100))
	return

filename = 'makeupCNN.txt'

model = testModel()


