import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse

sets=[('2007', 'trainfile'), ('2007', 'valfile')]


def parse_args():
    parser = argparse.ArgumentParser(description='Input File Path')
    parser.add_argument('--class_list', dest='class_list', help='create train txt',
                       default='pascal', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
# classes = ['Truck', 'Car', 'Licence_plate', 'Ashok_Leyland','Tata','Maruti_Suzuki','Bharat_Benz','Eicher','Other_Logo']
# classes=['aeroplane']  ### Define your own classes here

classes = []
z = args.class_list.split(',')
for i, item in enumerate(z):
   if i == 0:
       item = item.split('[')[1]
   item = item.split(']')[0]
   classes.append(item)
print(classes)

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    
    
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    obj_list = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text


        if cls not in classes:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        obj_list.append([str(cls_id), bb])

    if obj_list:
        out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        for obj in obj_list:
            bb= obj [1]
            out_file.write(obj[0] + " " + " ".join([str(a) for a in bb]) + '\n')
    
wd = os.getcwd()

list_file = open('full_data.txt', 'w')
for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    for image_id in image_ids:
        convert_annotation(year, image_id)
list_file.close()

with open('full_data.txt', 'r') as file:
    data = file.readlines()

print('[INFO]... Done with data refinement...!!')

train_file = open('2007_trainfile.txt','w')
val_file = open('2007_valfile.txt', 'w')
cnt = 0

print('[INFO]... Splitting the refined data...!!')
for points in data:
    cnt += 1
    if ((float(cnt) / float(len(data))) < 0.75):
        train_file.write(points)
    else:
        val_file.write(points)

train_file.close()
val_file.close()


os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")