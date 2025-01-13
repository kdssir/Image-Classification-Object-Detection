from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import statistics
import os, time

import tensorflow as tf
print("[INFO] Setting GPU usage...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print('GPU restriction Error',e)

print("[INFO] loading Mask model...")
print("[INFO] loading face mask detector model...")
model = load_model('./mask_detector_mafa_office_mixed.model')
datadir = './data_train'
classes = ['Masked', 'Unmasked']



def predict_mask(img):
	print("[INFO] predicting masks ...")
	face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)
	face = np.expand_dims(face, axis=0)
	(mask, withoutMask) = model.predict(face)[0]
	label = "Mask" if mask > withoutMask else "No Mask"
	return label

if __name__=='__main__':

	for dirs in os.listdir(datadir):
		print(dirs)
		if dirs in classes:
			dir_path = os.path.join(datadir, dirs)
			for imgs in os.listdir(dir_path):
				img_path = os.path.join(dir_path, imgs)
				print(img_path)
				img = cv2.imread(img_path)
				label = predict_mask(img)
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				img = cv2.resize(img,(300,300))
				cv2.putText(img, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				print("Prediction : "+label)
				# cv2.imshow("Prediction : "+label, img)
				# cv2.waitKey(0)
				# cv2.destroyWindow("Prediction : "+label)
				print('==========================================')
	print('avg inf time',statistics.mean(inf_time))
