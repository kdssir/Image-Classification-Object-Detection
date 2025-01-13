from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import BatchNormalization, Lambda, Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications.mobilenetv2 import MobileNetV2

def MobileNet(classes, img_size):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(img_size, img_size, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(classes, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    return model
