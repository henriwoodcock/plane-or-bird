#import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from sklearn.utils import shuffle
from plane_or_bird.data_clean import x_train, y_train, x_test, y_test
from plane_or_bird.data_generate import data_func
from plane_or_bird.data_generate import data_functions


data_aug_bool = input("would you like data_augmentation?")

#save a copy of original x_test before standardising:
x_test_norm = x_test
#normalise the data (could be standardised (between 0 and 1)):
full_data = np.append(x_train, x_test,axis=0)
mean = np.mean(full_data, axis=(1,2), keepdims=True)
std = np.std(full_data, axis=(1,2), keepdims=True)
full_data = (full_data - mean) / std
#reassign the training and testing data:
x_train = full_data[0:10000]
x_test = full_data[10000:12000]
#shuffle the training data:
x_train, y_train =data_functions.shuffle_data(x_train, y_train)

#set dense net variables:
input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
num_dense_blocks = 3 #can be changed, check densenet for meaning
num_bottleneck_layers = 5 #similarly
data_augmentation = True #data augmentation includes dropout etc.
num_classes=2 #output classes, birds or plane? is binary.
num_filters_bef_dense_block = 3 #init conv size
growth_rate = 12
compression_factor = 0.5 #neuron decrease each block
#create DenseNet:
inputs = keras.Input(shape=input_shape)
x = keras.layers.BatchNormalization()(inputs)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(num_filters_bef_dense_block,kernel_size=3, \
                        padding='same', kernel_initializer='he_normal')(x)
x = keras.layers.Concatenate()([inputs, x])
# stack of dense blocks bridged by transition layers
for i in range(num_dense_blocks):
    # a dense block is a stack of bottleneck layers
    for j in range(num_bottleneck_layers):
        y = keras.layers.BatchNormalization()(x)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(4 * growth_rate,kernel_size=1, \
                                padding = "same", \
                                kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = keras.layers.Dropout(0.2)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(growth_rate,kernel_size=3, padding = "same", \
                                kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = keras.layers.Dropout(0.2)(y)
        x = keras.layers.Concatenate()([x,y])
    # no transition layer after the last dense block
    if i == num_dense_blocks - 1:
        continue
    # transition layer compresses num of feature maps and
    # reduces the size by 2
    num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
    num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
    y = keras.layers.BatchNormalization()(x)
    y = keras.layers.Conv2D(4 * growth_rate,kernel_size=1, padding = "same", kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = keras.layers.Dropout(0.2)(y)
    x = keras.layers.AveragePooling2D()(y)

#add classifier on top
x = keras.layers.AveragePooling2D(pool_size=8)(x)
y = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid')(y)
# instantiate and compile model

# orig paper uses SGD but RMSprop works better for DenseNet
model = keras.Model(inputs=inputs, outputs=outputs)
#from keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.summary()

#fit model to data:
if data_aug_bool:
    model.fit_generator(data_func(x_train,y_train, 64), epochs=10, shuffle=True)
    model.evaluate(x_test,y_test, verbose = 0, batch_size =64)
    save = input("Would you like to save the weights? (y/n)")
    if save == "y":
        weights_name = input("Name the weights")
        model.save("/plane_or_bird/pretrained/" + str(weights_name) + "/.h5")
else:
    model.fit(x_train, y_train, epochs=10, batch_size=64, shuffle=True,validation_split=0.2)
    #evalute fit model against testing data
    model.evaluate(x_test,y_test, verbose = 0, batch_size =64)
    save = input("Would you like to save the weights? (y/n)")
    if save == "y":
        weights_name = input("Name the weights")
        model.save("/plane_or_bird/pretrained/" + str(weights_name) + "/.h5")
