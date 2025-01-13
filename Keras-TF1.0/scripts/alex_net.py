import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import ZeroPadding2D
from keras.layers import LeakyReLU
# import BatchNormalization
from keras.models import Model
from keras.layers import BatchNormalization, Lambda, Input, Conv3D, Dense, Conv2D,MaxPooling2D, Convolution2D, MaxPooling2D, MaxPooling3D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout


def AlexNet(CLASSES, IMAGE_SIZE):
    # Initialising the CNN
    classifier = Sequential()

    # First Convolution Block
    classifier.add(Conv2D(
        96, 11, strides=(4, 4), input_shape=(224, 224, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    classifier.add(BatchNormalization())

    # Second Convolution Block
    classifier.add(Conv2D(256, 5, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    classifier.add(BatchNormalization())

    # Third Convolution Block
    classifier.add(Conv2D(384, 3, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Fourth Convolution Block
    classifier.add(Conv2D(384, 3, strides=(1, 1),
                          activation='relu', padding='same'))

    # Fifth Convolution Block
    classifier.add(Conv2D(256, 3, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(Dropout(0.5))

    # Fully connected layer
    classifier.add(Flatten())

    # First hidden unit
    classifier.add(Dense(512, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.5))

    # Second hidden unit
    classifier.add(Dense(512, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.5))

    # Output layer
    classifier.add(Dense(CLASSES, activation='softmax',
                         kernel_initializer='uniform'))

    classifier.summary()
    return classifier

def get_liveness_model(num_classes, img_size, num_channels):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(img_size,img_size,num_channels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def livelyNess_model(num_classes, img_size, num_channels):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=(img_size, img_size, num_channels)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


def simple_CNN(num_classes):
    model = Sequential() 
    model.add(Flatten(input_shape=(200, 70, 3))) 
    model.add(Dense(100)) 
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5)) 
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3)) 
    model.add(Dense(num_classes, activation='softmax'))
    return model

def CNN_try2(num_classes):
    classifier=Sequential()
    classifier.add(Convolution2D(32,3,3,input_shape=(100,100,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    classifier.add(Convolution2D(64,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=64,activation='relu'))
    classifier.add(Dropout(p=0.5))
    classifier.add(Dense(num_classes,activation='sigmoid'))
    return classifier
