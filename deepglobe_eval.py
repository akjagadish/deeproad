import os
import numpy as np
import keras.backend as K

from keras.models import load_model

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

model = load_model('/home/brvanove/deepglobe/out/weights.hdf5', custom_objects={'precision': precision, 'recall': recall})

from PIL import Image

img = Image.open('/data/deepglobe/road/test/1392_sat.jpg')
img.show()

img_arr = np.asarray(img)
img_arr_batch = np.expand_dims(img_arr, axis=0)
print(img_arr_batch.shape)

y_pred = model.predict(img_arr_batch)
print(y_pred)
y_unmasked = np.zeros_like(y_pred)
y_unmasked[y_pred >= 0.5] = 255
y_unmasked_reshaped = y_unmasked.reshape((1024, 1024, -1))
print(y_unmasked_reshaped)
y_img = np.zeros((1024, 1024, 3))

for i in range(y_unmasked_reshaped.shape[0]):
    for j in range(y_unmasked_reshaped.shape[1]):
        y_img[i, j, 0] = y_unmasked_reshaped[i, j]
        y_img[i, j, 1] = y_unmasked_reshaped[i, j]
        y_img[i, j, 2] = y_unmasked_reshaped[i, j]

print(y_img.shape)
y_pil = Image.fromarray(y_img.astype('uint8'))
y_pil.show()
