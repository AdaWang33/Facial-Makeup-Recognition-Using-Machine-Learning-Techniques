import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import sys
import h5py
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout
from PIL import Image
import scipy.misc
import cv2
import pickle
import time

############ prepare input data
start_time = time.time()

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

x_train = np.reshape(im,(100,192*192))
x_test = np.reshape(im_t,(65,192*192))
x_train = x_train/255
x_test = x_test/255



################################ prepare vae
batch_size = 1
original_dim = 192*192
latent_dim = 2
intermediate_dim = 300
#intermediate_dim2 = 10
#intermediate_dim3 = 7
#intermediate_dim4 = 6
#intermediate_dim5 = 5
nb_epoch = 2000
epsilon_std = 1.0

############################### build vae
x = Input(batch_shape=(batch_size, original_dim))
# encoder part
h1 = Dense(intermediate_dim, activation='relu')(x)
h_drop = Dropout(0.2)(h1)
#h2 = Dense(intermediate_dim2, activation='relu')(h1)
#h3 = Dense(intermediate_dim3, activation='relu')(h1)
#h4 = Dense(intermediate_dim4, activation='relu')(h2)
#h5 = Dense(intermediate_dim5, activation='relu')(h3)
z_mean = Dense(latent_dim)(h_drop)
z_log_var = Dense(latent_dim)(h_drop)

# define z
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling)([z_mean, z_log_var])

# decoder layers
#decoder_h5 = Dense(intermediate_dim5, activation='relu')
#decoder_h4 = Dense(intermediate_dim4, activation='relu')
#decoder_h3 = Dense(intermediate_dim3, activation='relu')
#decoder_h2 = Dense(intermediate_dim2, activation='relu')
decoder_h1 = Dense(intermediate_dim, activation='relu')
decoder_drop = Dropout(0.2)
decoder_mean = Dense(original_dim, activation='sigmoid')

# decoder part
#h_decoded5 = decoder_h5(z)
#h_decoded4 = decoder_h4(h_decoded5)
#h_decoded3 = decoder_h3(h_decoded4)
#h_decoded2 = decoder_h2(h_decoded3)
h_decoded1 = decoder_h1(z)
h_decoded_drop = decoder_drop(h_decoded1)
x_decoded_mean = decoder_mean(h_decoded_drop)

# define loss
def vae_loss(x, x_decoded_mean):
#    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)/original_dim
    return xent_loss + kl_loss

# build vae which maps ocm to z
vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test,x_test))


# build encoder which projects inputs on the latent space
encoder = Model(x, z_mean)


# build decoder which samples from the learned distribution
decoder_input = Input(shape=(latent_dim,))
#_h_decoded5 = decoder_h5(decoder_input)
#_h_decoded4 = decoder_h4(_h_decoded5)
#_h_decoded3 = decoder_h3(_h_decoded4)
#_h_decoded2 = decoder_h2(_h_decoded3)
_h_decoded1 = decoder_h1(decoder_input)
_h_decoded_drop = decoder_drop(_h_decoded1)
_x_decoded_mean = decoder_mean(_h_decoded_drop)
generator = Model(decoder_input, _x_decoded_mean)
generator.save("generator_weights.hdf5")

#What we've done so far allows us to instantiate 3 models:
#an end-to-end autoencoder mapping inputs to reconstructions
#an encoder mapping inputs to the latent space
#a generator that can take points on the latent space and will output the corresponding reconstructed samples



####################### for the training data

x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


z_train = x_train_encoded
z_test = x_test_encoded


f = open('z_train', 'w')
pickle.dump(z_train, f)
f.close()

f = open('z_test', 'w')
pickle.dump(z_test, f)
f.close()

'''
# display a 2D plot of the digit classes in the latent space
t = np.arange(100)
s = 121
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.scatter(x_train_encoded[:, 0], x_train_encoded[:, 0] ,c=t)
d_train = ax.collections[0]
d_train.set_offset_position('data')
pos_train = d_train.get_offsets()
#print(pos_train[:,0])
plt.colorbar(ax.scatter(x_train_encoded[:, 0], x_train_encoded[:, 0] ,c=t))
plt.show()
# for validation purpose, refer to https://plot.ly/matplotlib/scatter/
'''

##################### for the testing part
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

'''
# display a 2D plot of the digit classes in the latent space
t = np.arange(65)
fig = plt.figure(figsize=(10,10))
ax_test = fig.add_subplot(1,1,1)
ax_test.scatter(x_test_encoded[:, 0], x_test_encoded[:, 0] ,c=t)
d_test = ax_test.collections[0]
d_test.set_offset_position('data')
pos_test = d_test.get_offsets()
plt.colorbar(ax_test.scatter(x_test_encoded[:, 0], x_test_encoded[:, 0] ,c=t))
plt.show()
'''
x_test_decoded = generator.predict(x_test_encoded)
#print(x_test_decoded.shape)

for i in range(65):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = x_test_decoded[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/recon_vae_test/" + fn + ".jpg" , img)
print("--- %s seconds ---" % (time.time() - start_time))




