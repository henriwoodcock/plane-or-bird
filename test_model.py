import tensorflow as tf
from data_clean import x_train, y_train, x_test, y_test
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


full_data = np.append(x_train, x_test,axis=0)

mean = np.mean(full_data, axis=(1,2), keepdims=True)
std = np.std(full_data, axis=(1,2), keepdims=True)
full_data = (full_data - mean) / std

x_train = full_data[0:5000]
x_test = full_data[5000:6000]


keep_prob = 0.25
model = keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Dropout(1-keep_prob))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(1-keep_prob))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(1-keep_prob))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train, y_train, nb_epoch=5, batch_size=1, shuffle=True)
