import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer



def create_training_data(traindatadir, categories, img_size, num_channels):


    print("[INFO].... Reading the Dataset...")
    training_data = []
    try:
        for catagory in categories:
            
            print(catagory)

            path = os.path.join(traindatadir, catagory) # path to cats or dogs
            class_num =  categories.index(catagory)

            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img))
                if num_channels ==  1:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append([new_array, class_num])

    except Exception as e:
        print(e.message)
        pass
    return  training_data

def feature_label_sep(training_data):

    print("[INFO].... Parsing the Dataset...")
    x = []
    y = []

    np.random.shuffle(training_data)

    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = np.array(y)

    return x,y

#test_proportion of 3 means 1/3 so 33% test and 67% train
def shuffle(matrix, target, test_proportion):
    ratio = matrix.shape[0]/test_proportion
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test


def parse_data(datadir, categories, num_classes, img_size, num_channels):


    training_data = create_training_data(datadir, categories, img_size, num_channels)
    x, y = feature_label_sep(training_data)

    print("[INFO].... Preparing the data for training...")
    category = []
    for i in range(num_classes):
        category.append(i)
    lb = LabelBinarizer()
    if num_classes > 2:
        y = lb.fit_transform(y)

    else:     
        y = lb.fit_transform(y)
        y = to_categorical(y)
    print(y)
    x = x / 255.0
    print(y)

    train_pct_index = int(0.7 * len(x))
    x_train, x_test = x[:train_pct_index], x[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    if num_channels ==1:
        a, _, _, = x_train.shape
        c = 1
        b, _, _ = x_test.shape
    else:
        a, _, _, c = x_train.shape
        b, _, _, _ = x_test.shape

    # Reshape the trained data
    x_train = x_train.reshape(a, img_size, img_size, c)
    
    y_train = y_train.reshape(a, num_classes)

    x_test = x_test.reshape(b, img_size, img_size, c)
    y_test = y_test.reshape(b, num_classes)

    return x_train, y_train, x_test, y_test
