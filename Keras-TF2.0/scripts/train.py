# python train_mask_detector.py --dataset dataset

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from generate_reports import gen_report, report_N_matrix
# from model import get_model, AlexNet_model
# from resnet50 import model_resnet50
from vgg16 import model_vgg16
from prepare_dataset import create_dataset

from tensorflow.python.client import device_lib

device_lib.list_local_devices()



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)


def model_training(num_classes, img_size, datadir, BS, EPOCHS, INIT_LR, weight_dir, save_results_to, categories,save_model_to):
    lb = LabelBinarizer()
    trainX, testX, trainY, testY = create_dataset(datadir, img_size, lb, categories)
    model = model_vgg16(img_size, num_classes)
    # model = model_resnet50 (img_size, num_classes)
    # model = AlexNet_model(img_size, num_classes)

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # compile our model
    print("[INFO] compiling model...")
    filepath = os.getcwd() + "/weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["acc"])

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        callbacks=callbacks_list,
        epochs=EPOCHS)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    y_pred = model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability

    print("[INFO] saving gender detector model...")
    model.save(save_model_to + 'mask_classifier.model', save_format="h5")
    model.save_weights(weight_dir + 'final_weights.h5')

    report_N_matrix(save_results_to, y_pred, testY, categories, '.png')
    gen_report(testY, y_pred, lb, H, EPOCHS, save_results_to)


if __name__ == '__main__':

    #datadir = os.getcwd() + '/data/mask_classifier_training'
    datadir = '/home/cocoslabs/alexnet_mask/tf_2.0/mask_classification_data/data/train/train'

    INIT_LR = 1e-7
    EPOCHS = 300
    BS = 32
    num_classes = 2
    img_size = 128

    save_results_to = os.getcwd() + '/tf_2.0/results/'
    save_model_to = os.getcwd() + '/tf_2.0/models/'
    save_weights_to = os.getcwd() + '/tf_2.0/weights/'
    categories = ['masked', 'unmasked']
    # categories = ['with_mask',  'without_mask']

    directories = [save_results_to, save_model_to, save_weights_to]
    for dirs in directories:
        if not os.path.exists(dirs):
            os.mkdir(dirs)

    model_training(num_classes, img_size, datadir, BS, EPOCHS, INIT_LR, save_weights_to, save_results_to, categories, save_model_to)
