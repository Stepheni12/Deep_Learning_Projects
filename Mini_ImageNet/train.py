import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

##### Read in data #####
x_train = np.load(sys.argv[1])
y_train = np.load(sys.argv[2])
model_name = sys.argv[3]

x_train = x_train.transpose((0,2,3,1))

##### Set model parameters #####
batch_size = 32 
nb_classes = 10
nb_epoch = 1 

img_channels = 3
img_rows = 112
img_cols = 112

##### Convert labels to one-hot matrix #####
y_train = to_categorical(y_train, nb_classes)

##### Create Model #####
model = Sequential()

#Block 1
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=[112, 112, 3]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.45))

#Block 2
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.45))
#6x6

#Block 3
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.45))

#Dense layer
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
#opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.01, decay=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


def train():
	model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True)

train()
model.save(model_name)
