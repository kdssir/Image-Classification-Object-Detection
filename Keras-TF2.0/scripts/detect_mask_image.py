from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import statistics
import os, time

print("[INFO] loading Mask model...")
print("[INFO] loading face mask detector model...")
model = load_model('./alexnet/mask_classify.model')
datadir = './alexnet/data_test'
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


def blur_detector(image):
	img = cv2.resize(image, dsize=(224, 224))
	im_cp = img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 150, 255, 0)
	kernel = np.ones((5, 5), np.uint8)
	dilated_img = cv2.dilate(thresh, kernel, iterations=1)
	x, y, _ = img.shape
	for i in range(0, x):
		for j in range(0, y):
			if dilated_img[i][j] == 255:
				im_cp[i][j][0] = 0
				im_cp[i][j][1] = 0
				im_cp[i][j][2] = 0
	im_gray = cv2.cvtColor(im_cp, cv2.COLOR_BGR2GRAY)
	edge = cv2.Canny(im_gray, 75, 150)
	edge_c = edge[27:190, 27:190]
	x, y = edge_c.shape
	count = 0
	for i in range(0, x):
		for j in range(0, y):
			if edge_c[i][j] == 255:
				count = count + 1
	return count

if __name__=='__main__':

	for dirs in os.listdir(datadir):
		print(dirs)
		if dirs in classes:
			cnt = 0
			dir_path = os.path.join(datadir, dirs)
			for imgs in os.listdir(dir_path):
				img_path = os.path.join(dir_path, imgs)
				print(img_path)

				img = cv2.imread(img_path)
				count = blur_detector(img)
				if count > 1200:
					cnt += 1
					print('Count', count)
					start = time.time()
					label = predict_mask(img)
					color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
					img = cv2.resize(img,(300,300))
					cv2.putText(img, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					print('inf time', time.time() -  start)
					print("Prediction : "+label)
					# cv2.imshow("Prediction : "+label, img)
					# cv2.waitKey(0)
					# cv2.destroyWindow("Prediction : "+label)
					print('==========================================')
					if cnt > 50:
						break
