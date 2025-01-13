from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import itertools
import cv2
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from prepare_dataset import parse_data
from alex_net import AlexNet, get_liveness_model, livelyNess_model,simple_CNN,CNN_try2
from vgg_16 import VGG_16
from mobile_net import MobileNet
from Generate_reports import acc_loss_analysis, report_N_matrix
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 56} )
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
# sess = tf.Session(config=config)
# K.set_session(sess)

print("Hello There.... \nLets get started....")


def model_training(datadir, categories, batch_size, nb_epoch,  num_classes, img_size, num_channels, model_dir, weight_dir, model_name):

    x_train, y_train, x_test, y_test =  parse_data(datadir, categories, num_classes, img_size, num_channels)

    filepath=os.getcwd() + "/weights/"+model_name+"_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # model = MobileNet(num_classes, img_size)
    model = AlexNet(num_classes, img_size)
    # model = simple_CNN(num_classes)
    #model = CNN_try2(num_classes)
    # model = livelyNess_model(num_classes, img_size, num_channels)
    # model = get_liveness_model(num_classes, img_size, num_channels)
    sgd = SGD(lr=0.0005, nesterov=True)
    opt = Adam(lr=1e-4, decay=1e-4 / 32)
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

    print("[INFO].... Training the Model...")

    model_var = model.fit(x_train, y_train,
                    # steps_per_epoch=16,
                        batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(x_test, y_test),
                       # validation_steps=30,
                         validation_split=0.3,
                        callbacks=callbacks_list
                        )

    # Save the trained model
    print("[INFO].... Saving the model....")
    model.save(model_dir + model_name+'_model.h5')

    # Save the weights
    print("[INFO].... Saving the weights....")
    model.save_weights(weight_dir + model_name+'final_weights.h5', overwrite=True)

    # Test the model
    print("[INFO].... Testing the Model....")
    y_pred = model.predict(x_test)

    return model_var, y_pred, y_test


if __name__=="__main__":


    # You can change the model parameters here..
    batch_size = 32
    img_size = 224
    nb_epoch = 20
    num_channels = 3

    # Dataset Directory
    # datadir = os.getcwd()+ '/fake_face_data/dataset/'
    datadir = './alexnet/Cat_n_Dog'
    categories = [ 'Cat', 'Dog']

    num_classes = len(categories)

    Extension = ".png"

    # Directories
    save_results_to = os.getcwd()+ '/results/'
    save_model_to = os.getcwd()+ '/models/'
    save_weights_to = os.getcwd()+ '/weights/'
    model_name = 'CNN_try2'

    directories = [save_results_to, save_model_to, save_weights_to]
    for dirs in directories:
        if not os.path.exists(dirs):
            os.mkdir(dirs)

    model_var, y_pred, y_test = model_training(datadir, categories, batch_size, nb_epoch,  num_classes, img_size, num_channels, save_model_to, save_weights_to, model_name)
    acc_loss_analysis(save_results_to, model_var, Extension)
    report_N_matrix(save_results_to, y_pred, y_test, categories,Extension)
