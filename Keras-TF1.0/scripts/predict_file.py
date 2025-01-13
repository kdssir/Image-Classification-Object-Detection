import cv2
from keras.models import load_model
import os, time
from subprocess import Popen, PIPE
import glob
import numpy as np


datadir = './alexnet/fake_face_data/dataset'
categories = [ 'fake', 'real']

img_size = 100


print("[INFO]... Loading the model...")
model = load_model('./alexnet/models/LivelyNess_Detection_try2_model.h5')

print("[INFO].... Loading the weights...")
model.load_weights('./alexnet/weights/LivelyNess_Detection_try2_weights-improvement-34-0.99.hdf5')

tot = 0
corr = 0
wro =0


for dirs in os.listdir(datadir):
    dir_path = './alexnet/fake_face_data/dataset/fake'
    tot += len(glob.glob1(dir_path, '*.jpg'))
    cnt = 0
    for imgs in os.listdir(dir_path):
        cnt += 1
        img_path = os.path.join(dir_path, imgs)
        print(img_path)
        start = time.time()
        img = cv2.imread(img_path)
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        img = np.array(img)


        img = img.reshape(1, img_size, img_size, 1)

        img = img / 255.0

        
        y_pred = model.predict(img)
        print('actual prediction',y_pred)
        print('inf time', time.time() -  start)

        y_pred_class = np.argmax(y_pred, axis=1)

        pred = categories[y_pred_class[0]]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 400))
        cv2.imshow('Prediction : '+ pred, img)
        cv2.waitKey(0)
        cv2.destroyWindow('Prediction : '+ pred)
        if cnt > 50:
            break
