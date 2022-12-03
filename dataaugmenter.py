import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import scipy.fft as fft
import cv2
import warnings

warnings.filterwarnings('ignore')

from glcm import *
from basics import *
from classifier import *

from keras.preprocessing.image import ImageDataGenerator

DRONES_DIR_GT = 'Data/Video_V/drones/gt/'
BIRDS_DIR_GT = 'Data/Video_V/birds/gt/'
DEFAULT_TRAJ = 'Data/trajectories_default/'
AUG_TRAJ = 'Data/trajectories_augmented/'

augmenter = ImageDataGenerator(        
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = 0.2)

augmented_imgs = augmenter.flow_from_directory(DEFAULT_TRAJ, class_mode = 'binary', batch_size=168,target_size=(432,432), seed = 42)[0]

i = 0
for img in augmented_imgs[0]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    folder = 'drone' if augmented_imgs[1][i]==1 else 'bird'
    name = AUG_TRAJ + folder + '/aug_' + str(i) + '.png'
    cv2.imwrite(name,img)
    i+=1