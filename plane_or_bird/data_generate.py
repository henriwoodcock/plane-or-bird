import tensorflow as tf
from tensorflow import keras
from plane_or_bird.data_clean import x_train, x_test, y_train, y_test

datagen = keras.preprocessing.image.ImageDataGenerator(\
featurewise_center = True, featurewise_std_normalization = True, \
rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, \
horizontal_flip=True)

datagen.fit(x_train)


def data_func(x_train, y_train, batch_size):
    datagen = keras.preprocessing.image.ImageDataGenerator(\
    featurewise_center = True, featurewise_std_normalization = True, \
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, \
    horizontal_flip=True)

    datagen.fit(x_train)

    return datagen.flow(x_train, y_train, batch_size = batch_size)
