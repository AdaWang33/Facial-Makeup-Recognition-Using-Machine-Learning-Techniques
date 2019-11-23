'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import matplotlib.image as mpimg
from scipy.stats import norm
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import sys
import h5py
K.set_image_dim_ordering('tf')
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout
from PIL import Image
import scipy.misc
import cv2
import pickle
import time



# input image dimensions
img_rows, img_cols, img_chns = 192, 192, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 1
original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 100

#<tf.Tensor 'input_1:0' shape=(1, 192, 192, 1) dtype=float32>
x = Input(batch_shape=(batch_size,) + original_img_size)
#<tf.Tensor 'conv2d_1/Relu:0' shape=(1, 192, 192, 1) dtype=float32>
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same',
                activation='relu')(x)
#<tf.Tensor 'conv2d_2/Relu:0' shape=(1, 96, 96, 64) dtype=float32>
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same',
                activation='relu',
                strides=(2, 2))(conv_1)
#<tf.Tensor 'conv2d_3/Relu:0' shape=(1, 96, 96, 64) dtype=float32>
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same',
                activation='relu',
                strides=1)(conv_2)
#<tf.Tensor 'conv2d_4/Relu:0' shape=(1, 96, 96, 64) dtype=float32>
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same',
                activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
#<tf.Tensor 'dense_1/Relu:0' shape=(?, 128) dtype=float32>
hidden = Dense(intermediate_dim, activation='relu')(flat)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`

#tf.Tensor 'lambda_1/add:0' shape=(?, 2) dtype=float32>
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 14, 14)
else:
    output_shape = (batch_size, 14, 14, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

#<tf.Tensor 'dense_4_2/Relu:0' shape=(?, 128) dtype=float32>
hid_decoded = decoder_hid(z)
#<tf.Tensor 'dense_5_2/Relu:0' shape=(?, 12544) dtype=float32>
up_decoded = decoder_upsample(hid_decoded)
#<tf.Tensor 'reshape_3_1/Reshape:0' shape=(?, 14, 14, 64) dtype=float32>
reshape_decoded = decoder_reshape(up_decoded)
#<tf.Tensor 'conv2d_transpose_10/Relu:0' shape=(?, ?, ?, 64) dtype=float32>
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
#<tf.Tensor 'conv2d_transpose_11/Relu:0' shape=(?, ?, ?, 64) dtype=float32>
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
#<tf.Tensor 'conv2d_transpose_9/Relu:0' shape=(?, ?, ?, 64) dtype=float32>
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
#<tf.Tensor 'conv2d_9/Sigmoid:0' shape=(?, ?, ?, 1) dtype=float32>
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean_squash)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()




# train the VAE on MNIST digits

# try to seperate MRI into training data and testing data
im = np.zeros((100,192,192))
im_t = np.zeros((65,192,192))
for i in range(100):
        fn = str(i)
        if i < 10:
          fn = '0'+fn
        im[i] = scipy.misc.imread("/home/ada/code/new_ori_vae_train/" + fn + ".jpg")

for i in range(65):
        fn = str(i)
        if i < 10:
          fn = '0'+fn
        im_t[i] = scipy.misc.imread("/home/ada/code/new_ori_vae_test/" + fn + ".jpg")

x_train = np.reshape(im,(100,192,192,1))
x_test = np.reshape(im_t,(65,192,192,1))
x_train = x_train/255
x_test = x_test/255



print('x_train.shape:', x_train.shape)

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

