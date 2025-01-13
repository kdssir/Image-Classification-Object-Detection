from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def AlexNet_model(img_size, num_classes):

    model =  Sequential()

    #Head
    # 1st convolution block
    model.add(Conv2D(96, 11, strides=(4,4), activation= 'relu', input_shape= (img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(3,3), strides= 2,))
    model.add(BatchNormalization())


    # 2nd Convolution Block
    model.add(Conv2D(256, 5, strides=(1,1), padding= 'same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides= 2)) 
    model.add(BatchNormalization())                   

    # 3rd Convlution Block
    model.add(Conv2D(384, 3, strides=(1,1), padding= 'same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # 4th Convolution Block
    model.add(Conv2D(384, 3, strides=(1,1),  padding= 'same', activation= 'relu'))

    # 5th Convolution Block
    model.add(Conv2D(256, 3, strides=(1,1), padding= 'same', activation= 'relu'))

    #Neck
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation= 'relu',kernel_initializer='uniform'))
    model.add(Dense(512, activation= 'relu',kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation= 'softmax',kernel_initializer='uniform'))

    return model



def get_model(img_size, classes ):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(img_size, img_size, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(classes, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
	    layer.trainable = False
    
    return model
