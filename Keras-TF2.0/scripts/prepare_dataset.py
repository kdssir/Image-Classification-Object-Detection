from imutils import paths
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

def create_dataset(dataDir, img_size, lb, categories):

    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataDir))
    data = []
    labels = []
    cnt = 0
    # loop over the image paths
    for imagePath in imagePaths:
        # print(os.getcwd() +'/'+ imagePath)
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        try:
            # load the input image (224x224) and preprocess it
            image = load_img(imagePath, target_size=(img_size, img_size))
            image = img_to_array(image)
            image = preprocess_input(image)
        except Exception as E:
            print(E)
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    print(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
        
    return trainX, testX, trainY, testY