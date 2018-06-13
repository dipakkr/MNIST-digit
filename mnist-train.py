from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Flatten, Dense
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 28, 28

# DATA PATH
train_dir = 'data/train'
validation_dir = 'data/validation'

# Parameters
train_samples = 60000
validation_samples = 10000
n_epochs = 30

# *** Model Begins ** #
model = Sequential()
model.add(Conv2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, 5, 5, activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
# ** Model Ends ** #

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        nb_epoch=n_epochs,
        validation_data=validation_generator,
        nb_val_samples=validation_samples)

model.save_weights('mnist-digit-custom.h5')
