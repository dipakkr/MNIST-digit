from keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ** plot 4 images ** #

# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
#
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
#
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.show()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


print (num_classes)


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
model.save_weights('mnist-digit.h5')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
