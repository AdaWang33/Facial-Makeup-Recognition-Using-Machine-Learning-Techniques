#/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pylab 
import matplotlib.image as mpimg
from scipy.stats import norm
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
import sys
import h5py
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import scipy.misc
import cv2
import pickle
from keras.optimizers import SGD
import time
from keras import optimizers
############################### load data
start_time = time.time()

# load OCM
hdf5_file_name = 'ocm_data.h5'
dataset_name   = '/ocm_data'
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
dataset = file[dataset_name]
ocm  = dataset[:] # first dimension is t, second dimension is T
file.close()
# convert the OCM data to float
ocm = ocm.astype('float64')
# envelope-detec and log the OCM signal
import os.path
if not(os.path.isfile("ocm_normalized.npy")):
    ocm = hilbert(ocm)
    ocm = np.log(ocm)
    ocm = np.abs(ocm)
    # save it so we don't have to perform this expensive operation every time
    np.save("ocm_normalized",ocm)
else:
    ocm = np.load("ocm_normalized.npy")
ocm = ocm - np.min(ocm)
ocm = ocm / np.max(ocm)

# crop the OCM data in little t direction, to remove clutter
ocm = ocm[3000:3100,:]

# prepare training data
N_OCM = 50
# the second dimension is the flatten vectors while the first to be number of vectors
x_train = np.zeros((6000,ocm.shape[0]*N_OCM))
for i in range(x_train.shape[0]):
	x_train[i,:] = ocm[:,i:i+N_OCM].flatten() 	
x_test = np.zeros((6000,ocm.shape[0]*N_OCM))
for i in range(x_test.shape[0]):
        x_test[i,:] = ocm[:,i+ x_train.shape[0]:i+ x_train.shape[0]+N_OCM].flatten()
x_all = np.zeros((12000,ocm.shape[0]*N_OCM))
for i in range(x_all.shape[0]):
        x_all[i,:] = ocm[:,i:i+N_OCM].flatten()

# prepare output,the correlation matrix of ocm
y_train = np.corrcoef(x_train)
y_test = np.corrcoef(x_test)
y_all = np.corrcoef(x_all)
y_train_real = y_train
y_test_real  = y_all[6000:,:6000]
scipy.misc.imsave('y_train_real.jpg', y_train_real)
scipy.misc.imsave('y_test_real.jpg', y_test_real)


print('y_train_real')
print(y_train_real.shape)

# fix random seed for reproducibility
np.random.seed(7)
# build model
model = Sequential()
model.add(Dense(4500, input_dim=5000, activation='relu'))
model.add(LeakyReLU(alpha=.001))   # add an advanced activation
model.add(Dropout(0.2))
model.add(Dense(3000, activation='relu'))
model.add(LeakyReLU(alpha=.001))   # add an advanced activation
model.add(Dropout(0.2))
model.add(Dense(6000, activation='linear'))
Adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=Adadelta, metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_train,y_train), epochs=10000, batch_size=100, verbose=2)

# make predictions
y_pred = model.predict(np.reshape(x_test[0,:],(1,ocm.shape[0]*N_OCM)))
y_train_pred = model.predict(x_train)
scipy.misc.imsave('y_train_pred.jpg', y_train_pred)
print('y_train_pred')
print(y_train_pred.shape)

y_test_pred  = model.predict(x_test) 
scipy.misc.imsave('y_test_pred.jpg', y_test_pred)

y_real = y_all[6000,0:6000]
y_real = np.reshape(y_real,(1,6000))

# make plot
x1 = np.arange(6000)
y1 = y_real[0,:]
y2 = y_pred[0,:]

'''
plt.figure(1)
plt.subplot(111)
plt.plot(x1,y1)
plt.show()
'''

pylab.plot(x1, y1, '-b', label='y_real')
pylab.plot(x1, y2, '-r', label='y_pred')
pylab.legend(loc='upper right')
pylab.ylim(-2.0, 2.0)
pylab.savefig('epoch_10000_num_6000_new.png')

#np.savetxt('y_pred',y_pred)
#np.savetxt('y_real',y_real)
#np.savetxt('y_sub',y_pred-y_real)




'''
#np.savetxt('y_train', y_train, delimiter=',')   # X is an array
#np.savetxt('y_pred', y_pred, delimiter=',')   # X is an array
scipy.misc.imsave('y_pred.jpg', y_pred)
scipy.misc.imsave('y_real.jpg',y_real)
scipy.misc.imsave('y_sub.jpg',np.subtract(y_pred,y_real))
'''
# create model
model.save('ocm_model.h5')  # creates a HDF5 file 'my_model.h5'

# evaluate the model
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("--- %s seconds ---" % (time.time() - start_time))


