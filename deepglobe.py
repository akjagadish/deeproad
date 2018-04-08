import os
import time
import numpy as np
import argparse
import random
import shutil
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint

from PIL import Image
from os import listdir
from os.path import isfile, join, splitext
from sklearn.metrics import f1_score

np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser()
parser.add_argument("NUM_EPOCHS", type=int, help="number of training epochs")
parser.add_argument("BATCH_SIZE", type=int, help="training batch size")
parser.add_argument("TRAIN_DIR", help="training directory")
parser.add_argument("VAL_DIR", help="validation directory")
parser.add_argument("OUT_DIR", help="output directory")
parser.add_argument("-p", "--prep", type=bool, help="flag to split train dir", default=False)
parser.add_argument("-s", "--seed", type=int, default=1)

args = parser.parse_args()

random.seed(args.seed)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024 * 1024, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print(model.summary())

def mask_binarizer(mask):
    mask_arr = np.array(mask)
    masks_mean_rgb = np.mean(mask_arr, axis=2)
    threshold_mask = np.zeros_like(mask_arr)
    threshold_mask[masks_mean_rgb < 128] = 0
    threshold_mask[masks_mean_rgb >= 128] = 255
    threshold_img = Image.fromarray(threshold_mask.astype('uint8'))
    return threshold_img 

def road_train_generator():
    data_gen_args = dict()

    mask_gen_args = dict(
        preprocessing_function=mask_binarizer)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    image_dir = join(args.TRAIN_DIR, 'X')
    mask_dir = join(args.TRAIN_DIR, 'y')
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        target_size=(1024, 1024),
        batch_size=args.BATCH_SIZE,
        class_mode=None,
        seed=args.seed)

    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=(1024, 1024),
        batch_size=args.BATCH_SIZE,
        class_mode=None,
        seed=args.seed)

    while 1:
        X = image_generator.next()
        y_mask = mask_generator.next()
        masks_max = np.max(y_mask, axis=3)
        y_softmax = np.zeros_like(masks_max)
        y_softmax[masks_max >= 128] = 1
        y_softmax[masks_max < 128] = 0
        del y_mask, masks_max
        y_softmax_flat = y_softmax.reshape(-1, y_softmax.shape[1] * y_softmax.shape[2])
        del y_softmax
        yield (X, y_softmax_flat)

def road_val_generator():
    mask_val_gen_args = dict(
        preprocessing_function=mask_binarizer)

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator(**mask_val_gen_args)

    image_dir = join(args.VAL_DIR, 'X')
    mask_dir = join(args.VAL_DIR, 'y')
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        target_size=(1024, 1024),
        batch_size=args.BATCH_SIZE,
        class_mode=None,
        seed=args.seed)

    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=(1024, 1024),
        batch_size=args.BATCH_SIZE,
        class_mode=None,
        seed=args.seed)

    while 1:
        X = image_generator.next()
        y_mask = mask_generator.next()
        masks_max = np.max(y_mask, axis=3)
        y_softmax = np.zeros_like(masks_max)
        y_softmax[masks_max >= 128] = 1
        y_softmax[masks_max < 128] = 0
        y_softmax_flat = y_softmax.reshape(-1, y_softmax.shape[1] * y_softmax.shape[2])
        yield(X, y_softmax_flat)

epoch_steps = len([f for f in listdir(join(args.TRAIN_DIR, 'X', 'img'))]) // args.BATCH_SIZE
valid_steps = len([f for f in listdir(join(args.VAL_DIR, 'X', 'img'))]) // args.BATCH_SIZE
outfile = join(args.OUT_DIR, 'weights.hdf5')
checkpointer = ModelCheckpoint(filepath=outfile, verbose=1, save_best_only=True)
model.fit_generator(road_train_generator(),
    steps_per_epoch=epoch_steps,
    epochs=args.NUM_EPOCHS,
    callbacks=[checkpointer], 
    validation_data=road_val_generator(),
    validation_steps=valid_steps,
    max_queue_size=1)    
