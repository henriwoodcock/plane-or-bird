import tensorflow as tf
from data_clean import x_train, y_train, x_test, y_test
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle

x_test_norm = x_test
y_test_norm = y_test

full_data = np.append(x_train, x_test,axis=0)

mean = np.mean(full_data, axis=(1,2), keepdims=True)
std = np.std(full_data, axis=(1,2), keepdims=True)
full_data = (full_data - mean) / std

x_train = full_data[0:10000]
x_test = full_data[10000:12000]


x_train, y_train = shuffle(x_train, y_train)
x_train, y_train = shuffle(x_train, y_train)
x_train, y_train = shuffle(x_train, y_train)
x_train, y_train = shuffle(x_train, y_train)


from tensorflow.keras.optimizers import RMSprop
# start model definition
# densenet CNNs (composite function) are made of BN-ReLU-Conv2D
input_shape=(32,32,3)
num_dense_blocks = 3
num_bottleneck_layers = 5
data_augmentation = True
num_classes=2
num_filters_bef_dense_block = 3
growth_rate = 12
compression_factor = 0.5
inputs = keras.Input(shape=input_shape)
x = keras.layers.BatchNormalization()(inputs)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(num_filters_bef_dense_block,kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
x = keras.layers.Concatenate()([inputs, x])
# stack of dense blocks bridged by transition layers
for i in range(num_dense_blocks):
    # a dense block is a stack of bottleneck layers
    for j in range(num_bottleneck_layers):
        y = keras.layers.BatchNormalization()(x)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(4 * growth_rate,kernel_size=1, padding = "same", kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = keras.layers.Dropout(0.2)(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)
        y = keras.layers.Conv2D(growth_rate,kernel_size=3, padding = "same", kernel_initializer='he_normal')(y)
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

model.fit(x_train, y_train, epochs=10, batch_size=64, shuffle=True,validation_split=0.2)
model.evaluate(x_test,y_test, verbose = 0, batch_size =64)

model.save("/plane_or_bird/pretrained/deepCNN.h5")

x_test, y_test, x_test_norm = shuffle(x_test, y_test, x_test_norm)


def print_pred_vs_real_and_image():
    num = np.random.randint(0,len(x_test))
    random_image = x_test[num]
    random_image = random_image.reshape(1,random_image.shape[0],random_image.shape[1],random_image.shape[2])
    yhat = model.predict(random_image)[0][0]
    if yhat > 0.5:
        y_hat = 1
    else:
        y_hat = 0
    if y_hat == 0:
        yhat_label = "plane"
    else:
        yhat_label = "bird"
    y = y_test[num]
    if y == 0:
        y_label = "plane"
    else:
        y_label = "bird"
    fig = plt.figure()
    plt.imshow(x_test_norm[num])
    fig.suptitle("Predicted = " + yhat_label + ", actual = " + y_label)
    fig.show()

#random_image = random_image.reshape(1,random_image.shape[0],random_image.shape[1],random_image.shape[2])
#model.predict(random_image).argmax()
correct = 0
for i in range(len(y_test)):
    image = x_test[i].reshape(1,x_test[i].shape[0],x_test[i].shape[1],x_test[i].shape[2])
    y_hat = model.predict(image)
    if y_hat > 0.5:
        y_hat = 1
    else:
        y_hat = 0
    y = y_test[i]
    if y == y_hat:
        cel = "matching"
        correct +=1
    else:
        cel = "not matching"
    print(y,y_hat,cel)

print(correct/len(y_test))
