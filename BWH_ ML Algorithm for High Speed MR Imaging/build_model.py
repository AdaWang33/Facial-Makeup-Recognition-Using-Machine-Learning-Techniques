#/usr/bin/env python
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
import h5py
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
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
############################### load data
start_time = time.time()

# import z(from latent space) in vae.py
f = open('z_train', 'r')
z_train = pickle.load(f)
f.close()
Y_train = z_train

# import z(from latent space) in vae.py
f = open('z_test', 'r')
z_test = pickle.load(f)
f.close()
Y_test = z_test



# load OCM
hdf5_file_name = 'ocm_data.h5'
dataset_name   = '/ocm_data'
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
dataset = file[dataset_name]
ocm  = dataset[:] # first dimension is t, second dimension is T
file.close()
# convert the OCM data to float
ocm = ocm.astype('float64')
# crop the OCM data in little t direction, to remove clutter
ocm = ocm[500:,:]
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

# load MRI
hdf5_file_name = 'mri_data.h5'
dataset_name   = '/real' # we look at the real values and ignore the comples part for now
file = h5py.File(hdf5_file_name, 'r') # 'r' means that hdf5 file is open in read-only mode
data = file[dataset_name]
mri  = data[:] # first dimension is T, second dimension are the planes, third and fourth dimensions are image axes
file.close()
# load us2mr
hdf5_file_name = 'mr2us.h5'
dataset_name   = '/plane1' # There is also /plane2 for the second mri plane
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
data = file[dataset_name]
mr2us  = data[:] # first dimension is T, second dimension are the planes, third and fourth dimensions are image axes
file.close()

# to reduce number of ocm traces to 100
mr2us = mr2us[0:165]
# sometimes there is more MRI data than OCM, so we need to remove some MRI images
if ocm.shape[0] < mr2us[len(mr2us)-1]:
    inds = np.where(mr2us > ocm.shape[0])
    mr2us = np.delete(mr2us,inds)
    mri = mri[0:mr2us.shape[0],:,:,:]
im = mri[1,1,:,:] # first image, first plane
# normalize to [0,1]
im = im - np.min(im)
im = im / np.max(im)

#try to rank ocm by the standard deviation of every row
dev = ocm.std(1)
nHighestStd = 1000
indices = np.argsort(dev)[::-1][:nHighestStd]
ocm = ocm[indices,:]

N_OCM = 60  # 60 OCM traces per MR image
ocm = ocm - ocm.min()
ocm = ocm / ocm.max()

# prepare ocm for input
X = np.ones(((mr2us.shape[0],1,ocm.shape[0],N_OCM)))
count = 0
for t in mr2us:
   t = t-1 # python indexing is 0-based
   X[count,0,:,:] = ocm[:,range(t-N_OCM,t)]
   count+=1
X_train = X[0:100]
X_test = X[100:165]

# prepare all ocm data
N_mri = 20000-N_OCM
steps = 10
X = np.ones(((N_mri/steps,1,ocm.shape[0],N_OCM)))
count = 0
for t in range(0,N_mri,steps):
   X[count,0,:,:] = ocm[:,range(t,t+N_OCM)]
   count+=1
X_all = X

#normalize
for i in range(0,100):
    X_train[i,0,:,:] = X_train[i,0,:,:] - float(np.amin(X_train[i,0,:,:]))
    X_train[i,0,:,:] = X_train[i,0,:,:] / float(np.amax(X_train[i,0,:,:]))

for i in range(0,65):
    X_test[i,0,:,:] = X_test[i,0,:,:] - float(np.amin(X_test[i,0,:,:]))
    X_test[i,0,:,:] = X_test[i,0,:,:] / float(np.amax(X_test[i,0,:,:]))


###################### build model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(32, 1, 1, border_mode='valid', input_shape=(1,nHighestStd,N_OCM), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(1,1)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(200, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='linear'))
	sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        # Compile model
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
print(Y_train)
model.fit(X_train, Y_train, validation_data=(X_train, Y_train), nb_epoch=30, batch_size=5, verbose=2)
#save the model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# Final evaluation of the model
scores = model.evaluate(X_train, Y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# generate Y_test
Y_train_predict = model.predict(X_train)
Y_test_predict = model.predict(X_test)
Y_all_predict = model.predict(X_all)

# display a 2D plot of the digit classes in the latent space
fig, ax = plt.subplots()
s = 121
train = ax.scatter(Y_train[:,0], Y_train[:,1], color='r', s=2*s,alpha=.4)
train_predict = ax.scatter(Y_train_predict[:,0], Y_train_predict[:,1], color='g', s=2*s,alpha=.4)
ax.legend((train,train_predict),
           ('train', 'train_predict'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
plt.show()

# display a 2D plot of the digit classes in the latent space
fig, ax = plt.subplots()
s = 121
train = ax.scatter(Y_train[:,0], Y_train[:,1], color='r', s=2*s,alpha=.4)
test_predict = ax.scatter(Y_test_predict[:,0], Y_test_predict[:,1], color='b', s=2*s,alpha=.4)
ax.legend((train,test_predict),
           ('train', 'test_predict'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
plt.show()

################################### decoder part
batch_size = 1
original_dim = 192*192
latent_dim = 2
intermediate_dim = 50
nb_epoch = 10
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h1 = Dense(intermediate_dim, activation='relu')(x)
h_drop = Dropout(0.2)(h1)
z_mean = Dense(latent_dim)(h_drop)
z_log_var = Dense(latent_dim)(h_drop)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
#    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                              std=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling)([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h1 = Dense(intermediate_dim, activation='relu')
decoder_drop = Dropout(0.2)
decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded1 = decoder_h1(z)
h_decoded_drop = decoder_drop(h_decoded1)
x_decoded_mean = decoder_mean(h_decoded_drop)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

#vae.fit(x_train, x_train,
#        shuffle=True,
#        nb_epoch=nb_epoch,
#        batch_size=batch_size,
#        validation_data=(x_train,x_train))

# build a model to project inputs on the latent space
#encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)
generator = load_model('/home/ada/code/generator_weights.hdf5')

# generate mri from Y
mri_train = generator.predict(Y_train)
mri_test = generator.predict(Y_test)
mri_train_predict = generator.predict(Y_train_predict)

mri_test_predict = generator.predict(Y_test_predict)
#print("Starting test set prediction\n")
#t0= time.clock()
mri_all_predict = generator.predict(Y_all_predict)
#t= time.clock() 
#print("done\n")
#print("took $t seconds\n")

# save mri
for i in range(100):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = mri_train[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/mri_train/" + fn + ".jpg" , img)

for i in range(65):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = mri_test[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/mri_test/" + fn + ".jpg" , img)


for i in range(100):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = mri_train_predict[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/mri_train_predict/" + fn + ".jpg" , img)

for i in range(65):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = mri_test_predict[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/mri_test_predict/" + fn + ".jpg" , img)


# save mri
for i in range(mri_all_predict.shape[0]):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    if i <100:
      fn = '0'+fn
    if i <1000:
      fn = '0'+fn
    img = mri_all_predict[i,:]
    img = np.reshape(img,(192,192))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
#    if img.mode != 'RGB':
#      img = img.convert('RGB')

    scipy.misc.imsave("/home/ada/code/mri_all_predict/" + fn + ".jpg" , img)
print("--- %s seconds ---" % (time.time() - start_time))


