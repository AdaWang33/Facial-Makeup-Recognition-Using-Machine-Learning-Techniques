# -*- coding: utf-8 -*-

from keras.preprocessing import image
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense




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
img_path = 'C:\\Users\\mohan_000\\Desktop\\2017fall\\MachineLearning\\project\\project_code\\New folder\\'#asianLabed\\
data, label, imgNum = loadImage(img_path)



# fix random seed for reproducibility
np.random.seed(7)
# create model
# We can piece it all together by adding each layer
model = Sequential()
# the first level has 12 nuerons and 8 inputs. Use relu function to deal with its output
model.add(Dense(12, input_dim=8, activation='relu'))
# the second level of CNN has 8 nuerons
model.add(Dense(8, activation='relu'))
# the last level has only 1 neuron to classify the output, to transfer its input into [0,1]
model.add(Dense(1, activation='sigmoid'))

# Compile model
#use logarithmic loss (log loss) for a binary classification
# use the efficient gradient descent algorithm “adam”
# collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
# epochs is the time of training. Iteration time is fixed on 150 times. Each instance will be passed by 150 times.
# weight will update after training 10 instances
# input is X and true label is Y
model.fit(data, label, epochs=150, batch_size=10)


# evaluate the model
scores = model.evaluate(data, label)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))