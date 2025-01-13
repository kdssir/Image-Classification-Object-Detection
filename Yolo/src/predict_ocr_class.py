from ctypes import *
import math, statistics
import random
import os
import cv2
import numpy as np
import time
import darknet
import scipy.spatial.distance as distance

class predict_ocr():

    netMin = metamain = netMin2 =  metamain2 = None

    def __init__(self):

        global netMin, metamain, netMin2, metamain2
        configPath = "./2MS_YV4/Model1/yolo-obj.cfg"
        weightPath = "./2MS_YV4/Model1/yolo-obj_best.weights"
        metaPath = "./2MS_YV4/Model1/obj.data"
        netMin, metamain = self.load_model(configPath, weightPath, metaPath)

        configPath = "./2MS_YV4/Model2/yolo-obj.cfg"
        weightPath = "./2MS_YV4/Model2/yolo-obj_best.weights"
        metaPath = "./2MS_YV4/Model2/obj.data"
        netMin2, metamain2 = self.load_model(configPath, weightPath, metaPath)

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def find_iou(self, boxA, boxB):
        
        boxA[:] = self.convertBack(boxA[0],boxA[1],boxA[2],boxA[3])
        boxB[:] = self.convertBack(boxB[0],boxB[1],boxB[2],boxB[3])

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        
        if interArea == 0:
            return 0
        
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def create_pairs(self, items):
        pairs = [(items[i],items[j]) for i in range(len(items)) for j in range(i+1, len(items))]
        return pairs

    def remove_duplication(self, b_box):
        # print(b_box)
        print('before b_box', len(b_box))
        print('=============================')
        toBe_del = []
        pairs = self.create_pairs(b_box)
        for pair in pairs:
            boxA = pair[0]
            boxB = pair[1]
            iou = self.find_iou(boxA[:-2], boxB[:-2])
            if iou > 0.45:
                probs = [boxA[5], boxB[5]]
                if probs.index(min(probs)) == 0:
                    ind = b_box.index(boxA)
                    if ind not in toBe_del:
                        toBe_del.append(ind)
                else:
                    ind = b_box.index(boxB)
                    if ind not in toBe_del:
                        toBe_del.append(ind)

                print('duplication deleted, found iou', iou)
                strng = '(' + str(boxA[5]) + '  ' + boxA[4] + ')' + ',' + '(' + str(boxB[5]) + '  ' + boxB[4] + ')'
                print('Probabilities of the iou', strng)
        for i, ind in enumerate(toBe_del):
            del b_box[ind - i]
        print('=============================')
        print('after b_box', len(b_box))
        return b_box

    def sorting_bounding_box(self, points):
        points = list(map(lambda x: [x[0], x[1][0], x[1][2]], points))
        # print(points)
        points_sum = list(map(lambda x: [x[0], x[1], sum(x[1]), x[2][1]], points))
        x_y_cordinate = list(map(lambda x: x[1], points_sum))
        final_sorted_list = []
        while True:
            try:
                new_sorted_text = []
                initial_value_A = [i for i in sorted(enumerate(points_sum), key=lambda x: x[1][2])][0]
                #         print(initial_value_A)
                threshold_value = abs(initial_value_A[1][1][1] - initial_value_A[1][3])
                threshold_value = (threshold_value / 2) + 5
                del points_sum[initial_value_A[0]]
                del x_y_cordinate[initial_value_A[0]]
                #         print(threshold_value)
                A = [initial_value_A[1][1]]
                K = list(map(lambda x: [x, abs(x[1] - initial_value_A[1][1][1])], x_y_cordinate))
                K = [[count, i] for count, i in enumerate(K)]
                K = [i for i in K if i[1][1] <= threshold_value]
                sorted_K = list(map(lambda x: [x[0], x[1][0]], sorted(K, key=lambda x: x[1][1])))
                B = []
                points_index = []
                for tmp_K in sorted_K:
                    points_index.append(tmp_K[0])
                    B.append(tmp_K[1])
                dist = distance.cdist(A, B)[0]
                d_index = [i for i in sorted(zip(dist, points_index), key=lambda x: x[0])]
                new_sorted_text.append(initial_value_A[1][0])

                index = []
                for j in d_index:
                    new_sorted_text.append(points_sum[j[1]][0])
                    index.append(j[1])
                for n in sorted(index, reverse=True):
                    del points_sum[n]
                    del x_y_cordinate[n]
                final_sorted_list.append(new_sorted_text)
            except Exception as e:
                print(e)
                break

        return final_sorted_list

    def sorting_data_creation(self, b_box):
        new_box =  []
        for box in b_box:
            xmin, ymin, xmax, ymax = self.convertBack(box[0], box[1], box[2], box[3])
            if xmax > 320:
                xmax = 320
            if ymax > 320:
                ymax = 320
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            new_box.append([box[4], [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]])
        return new_box

    def label_retrive(self, b_box):
        
        b_box = self.remove_duplication(b_box)
        label_str = ''
        conf_score = 0.0
        prob = []
        if b_box:
            for box in b_box:
                prob.append(box[5])

            try:
                conf_score = statistics.mean(prob)
            except:
                conf_score = 0.0
            new_box = self.sorting_data_creation(b_box)
            # print('new_box', new_box)
            sorted_arry = self.sorting_bounding_box(new_box)
            print('sorted array', sorted_arry)
            for list in sorted_arry:
                s = ("".join(list))
                label_str += s

        return label_str, conf_score


    def cvDrawBoxes(self, detections, img, b_boxes):

        for detection in detections:
            x, y, w, h = detection[2][0], \
                        detection[2][1], \
                        detection[2][2], \
                        detection[2][3]
            label = detection[0].decode('utf8')
            prob = round(detection[1] * 100, 4)
            box = [x, y, w, h, label, prob]
            b_boxes.append(box)
            # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
            # cv2.putText(img,
            #             detection[0].decode(),
            #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             [0, 255, 0], 2)
        return b_boxes


    def load_model(self, configPath, weightPath, metaPath):

        netMain = None
        metaMain = None
        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
        
        return netMain, metaMain


    def predit_withModel(self, netMain, metaMain, img, b_boxes):
        darknet_image = darknet.make_image(darknet.network_width(netMain),
                                        darknet.network_height(netMain), 3)
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)

        b_boxes = self.cvDrawBoxes(detections, img, b_boxes)
        return b_boxes



    def plate(self, frame_read):
        global netMin, metamain, netMin2, metamain2
        print('original img shape>>>>>>>>>>>>>', frame_read.shape)
        frame_read = cv2.resize(frame_read, (320, 320))
        frame_read = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        b_boxes = []
        b_boxes = self.predit_withModel(netMin, metamain, frame_read, b_boxes)
        b_boxes = self.predit_withModel(netMin2, metamain2, frame_read, b_boxes)
        label, conf = self.label_retrive(b_boxes)

        return label, conf



if __name__ == '__main__':
    des_path='./data/org_data/merged/pred/'
    dirs = ['./testing/lp_1','./data/org_data/merged/test_data/']
    for folder_path in dirs:
        inp_dir = './Pictures/lp_1'
        cnt = 0
        for i,img in enumerate(os.listdir(inp_dir)):
            cnt += 1
            b_boxes = []
            img_path = os.path.join(inp_dir, img)
            lp_img = lp_detect(img_path)
            # frame_read = cv2.imread(img_path)
            label, conf = plate(lp_img)
            cv2.imshow(label, lp_img)
            cv2.waitKey(0)
            cv2.destroyWindow(label)



