import os

os.system("cp cfg/yolov4-custom.cfg cfg/yolo-obj.cfg")

os.system("cp cfg/voc.data data/obj.data")

os.system("rm -rf /data/obj")

if not os.path.exists(os.getcwd() + '/data/obj'):
    os.mkdir(os.getcwd() + '/data/obj')


if not os.path.exists(os.getcwd() + '/VOCdevkit/VOC2007/ImageSets'):
    os.mkdir(os.getcwd() + '/VOCdevkit/VOC2007/ImageSets')

if not os.path.exists(os.getcwd() + '/VOCdevkit/VOC2007/ImageSets/Main'):
    os.mkdir(os.getcwd() + '/VOCdevkit/VOC2007/ImageSets/Main')

