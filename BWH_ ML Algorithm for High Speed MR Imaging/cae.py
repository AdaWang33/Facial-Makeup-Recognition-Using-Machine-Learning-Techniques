from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from time import sleep
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
'''
from scipy.signal import hilbert
'''
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout
from PIL import Image

from keras.layers import Flatten
from keras import backend as K
K.set_image_dim_ordering('tf')

import tensorflow as tf
#pre_trained_layers = []

input_img = Input(shape=( 192, 192, 1))

# build encoder
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

encoder = Model(input_img, encoded)
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#print(encoded)
# now output.shape == (24,24,1)

# build decoder 
x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
# now output.shape == (192,192,1)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# prepare input data
# load MRI
hdf5_file_name = 'mri_data.h5'
dataset_name   = '/real' # we look at the real values and ignore the comples part for now
file = h5py.File(hdf5_file_name, 'r') # 'r' means that hdf5 file is open in read-only mode
data = file[dataset_name]
mri  = data[:] # first dimension is T, second dimension are the planes, third and fourth dimensions are image axes
file.close()
import os.path
ocm = np.load("ocm_normalized.npy")
# load us2mr
hdf5_file_name = 'mr2us.h5'
dataset_name   = '/plane1' # There is also /plane2 for the second mri plane
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
data = file[dataset_name]
mr2us  = data[:] # first dimension is T, second dimension are the planes, third and fourth dimensions are image axes
file.close()

# sometimes there is more MRI data than OCM, so we need to remove some MRI
if ocm.shape[0] < mr2us[len(mr2us)-1]:
    inds = np.where(mr2us > ocm.shape[0])
    mr2us = np.delete(mr2us,inds)
    mri = mri[0:mr2us.shape[0],:,:,:]
print (mri.shape)

# try to seperate MRI into training data and testing data

#normalize
im = np.zeros((100,192,192))
for i in range(0,100):
    im[i] = mri[i,1,:,:]
    im[i] = im[i] - np.min(im[i])
    im[i] = im[i] / np.max(im[i])
im_t = np.zeros((67,192,192))
for i in range(100,167):
    im_t[i-100] = mri[i,1,:,:]
    im_t[i-100] = im_t[i-100] - np.min(im_t[i-100])
    im_t[i-100] = im_t[i-100] / np.max(im_t[i-100])


#x_train = x_train.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
im = np.reshape(im,(100,192,192,1))
im_t = np.reshape(im_t,(67,192,192,1))

from keras.callbacks import TensorBoard


autoencoder.fit(im,im,
            nb_epoch=30,
            batch_size=1,
            shuffle=True,
            validation_data=(im_t, im_t),
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
decoded_imgs = autoencoder.predict(im)

encoded_imgs = encoder.predict(im)

#print('encoded_imgs.shape')
#print(encoded_imgs.shape)


# display part
import matplotlib.pyplot as plt
# display reconstructed images and compresses representation
n = 10
plt.figure(figsize=(100, 100))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(im[i].reshape(192, 192))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(192, 192))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# display compressed representation
n = 10
plt.figure(figsize=(4, 3))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(48, 48 * 1).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

