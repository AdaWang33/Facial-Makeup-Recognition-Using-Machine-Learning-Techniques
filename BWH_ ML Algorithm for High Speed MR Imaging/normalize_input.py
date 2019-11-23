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

for i in range(100):
    fn = str(i)
    if i < 10:
      fn = '0'+fn
    img = cv2.imread("/home/frank/code/ori_vae_train/" + fn + ".jpg",0)
    equ = cv2.equalizeHist(img)
    equ = equ/1.75
    cv2.imwrite("/home/frank/code/new_ori_vae_train/" + fn + ".jpg",equ)

for i in range(100,165):
    fn = str(i)
    xin = str(i-100)
    if i-100<10:
      xin = '0'+xin
    img = cv2.imread("/home/frank/code/ori_vae_test/" + fn + ".jpg",0)
    equ = cv2.equalizeHist(img)
    equ = equ/1.75
    cv2.imwrite("/home/frank/code/new_ori_vae_test/" + xin + ".jpg",equ)

