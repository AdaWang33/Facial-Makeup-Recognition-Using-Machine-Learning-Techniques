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
from keras.preprocessing import image
import numpy as np

# Image size
width, height = 200, 140




def makeupPredict(filePath_makeupPic, filePath_pretrainedModel):
    img = image.load_img(filePath_makeupPic, target_size=(width,height))   
    #transform PIL image into array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    model_pred = load_model(filePath_pretrainedModel)
    
    preds = model_pred.predict_classes(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    if preds[0] == 0:
        result = 'without makeup'
    else:
        result = 'makeup'
    return result



filePath_makeupPic = '1.jpg'
filePath_pretrainedModel = 'makeupCNN.h5'

result = makeupPredict(filePath_makeupPic, filePath_pretrainedModel)

print(result)



