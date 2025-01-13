from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def model_resnet50(img_size,num_classes):
    model=Sequential()
    model.add(ResNet50(input_shape=(img_size,img_size,3), input_tensor=None,include_top=False,weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(540, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_resnet18(img_size,num_classes):
    model=Sequential()
    model.add(ResNet18(input_shape=(img_size,img_size,3), input_tensor=None,include_top=False,weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(540, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model