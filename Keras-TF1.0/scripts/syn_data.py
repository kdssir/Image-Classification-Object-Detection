from imgaug import augmenters as iaa
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

import shutil

def multiply(img, out_file_path,img_count):
    seq = iaa.Sequential([
        iaa.Multiply((1.25))
    ])
    img = cv2.imread(img)
    o_img = img

    images_aug = seq.augment_images(img)

    images_out = np.concatenate((o_img, images_aug), axis=1)
    cv2.imwrite(out_file_path + str(img_count) + 'multiply' + '.jpg', images_aug)


def g_blur(img, out_file_path,img_count):

    img = cv2.imread(img)
 
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(out_file_path + str(img_count) + 'g_blur' + '.jpg', blur)


def avg_blur(img, out_file_path,img_count):
   
    img = cv2.imread(img)
    kernel = np.ones((3, 3), np.float32) / 9
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(out_file_path + str(img_count) + 'avg_blur' + '.jpg', dst)

def rotate_image(img_path,out_file_path, img_count, deg):
    cont = 1
    image = cv2.imread(img_path)
    for i in range (deg[0], deg[1]):
        row, col, _ = image.shape
        center = tuple(np.array([row, col]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, i, 1.0)
        new_image = cv2.warpAffine(image, rot_mat, (col, row))
        cv2.imwrite(out_file_path + str(img_count)+'rotated_'+str(i)+'.jpg',new_image)
        print('img rotated')
        cont += 1


def add(img, out_file_path,img_count):
    seq = iaa.Sequential([
        iaa.Add((45), per_channel=0.5),

    ])

    img = cv2.imread(img)
    o_img = img
    color = img


    images_aug = seq.augment_images(color)

    images_out = np.concatenate((o_img, images_aug), axis=1)
    cv2.imwrite(out_file_path + str(img_count) + 'add' + '.jpg', images_aug)


def filter_color1(img, out_file_path,img_count):
    img = cv2.imread(img)
    color = img

    height, width, channels = color.shape
    b, g, r = cv2.split(color)

    rgb_split = np.empty([height, width, 3], 'uint8')
    rgb_split[:, 0:width] = cv2.merge([g, r, r])

    out_img = np.concatenate((img, rgb_split), axis=1)
    cv2.imwrite(out_file_path + str(img_count) + 'color1' + '.jpg', rgb_split)


def filter_color2(img, out_file_path,img_count):
    img = cv2.imread(img)
    color = img
    height, width, channels = color.shape
    b, g, r = cv2.split(color)

    rgb_split = np.empty([height, width, 3], 'uint8')
    rgb_split[:, 0:width] = cv2.merge([g, g, r])

    out_img = np.concatenate((img, rgb_split), axis=1)
    cv2.imwrite(out_file_path + str(img_count) + 'color2' + '.jpg', rgb_split)



def filter_color3(img, out_file_path,img_count):
    img = cv2.imread(img)
    color = img

    height, width, channels = color.shape
    b, g, r = cv2.split(color)

    rgb_split = np.empty([height, width, 3], 'uint8')
    rgb_split[:, 0:width] = cv2.merge([b, r, r])

    out_img = np.concatenate((img, rgb_split), axis=1)
    cv2.imwrite(out_file_path + str(img_count) + 'color3' + '.jpg', rgb_split)


count = 0

data_path ="./alexnet/fake_face_data/dataset/fake/"
out_path ="./alexnet/fake_face_data/dataset/fake_aug/"

import glob

img_count=0
temp_out_path = out_path 
        
for img_path in glob.glob(data_path+"/*"):
    print("image count", img_count)
    # if img_count<25:

        # print("image_path", img_path)
    g_blur(img_path, temp_out_path,img_count)
 
    rotate_image(img_path, temp_out_path, img_count, [-3,0])
    shutil.copy(img_path,temp_out_path)


    img_count += 1