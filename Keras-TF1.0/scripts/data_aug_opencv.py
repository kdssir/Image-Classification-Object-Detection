import cv2
import glob
import os
import shutil

# Folder_name="./Desktop/LPR/brand/dataset_aug/others/"
Extension=".jpg"

def resize_image(image,w,h, ind, Folder_name):
    image=cv2.resize(image,(w,h))
    cv2.imwrite(Folder_name+"/Resize-"+str(w)+"*"+str(h)+str(ind)+Extension, image)

#crop
def crop_image(image,y1,y2,x1,x2, ind, Folder_name):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/Crop-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+str(ind)+Extension, image)

def padding_image(Folder_name, image,ind, topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Folder_name + "/padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(rightBorder)+str(ind) + Extension, image)

def flip_image(image,dir, ind, Folder_name):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name + "/flip-" + str(dir)+str(ind)+Extension, image)

#
# def create_data(data_path, img_size):
#     count = 0
#
#     for dir in os.listdir(data_path):
#         directory1 = data_path + dir + '/'
#         temp_out_path = out_path + str(dir)
#
#         temp_out_path = temp_out_path + "/"
#
#         for img_path in glob.glob(directory1 + "/*.jpg"):
#             class_num = imgs.index(images)
#             img = cv2.imread(img_path)
#             image = cv2.resize(img, (img_size,img_size))
#             print("INFO..... Image is getting Processed....")
#             resize_image(image, 450, 400, class_num)
#             crop_image(image, 100, 400, 0, 350, class_num)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             crop_image(image, 100, 400, 100, 450, class_num)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             crop_image(image, 0, 300, 0, 350, class_num)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             # crop_image(image, 0, 300, 100, 450, class_num)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             # crop_image(image, 100, 300, 100, 350, class_num)  # (y1,y2,x1,x2)(bottom,top,left,right)
#
#             padding_image(image,class_num, 100, 0, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             padding_image(image, class_num, 0, 100, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             padding_image(image, class_num,0, 0, 100, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             padding_image(image, class_num,0, 0, 0, 100)  # (y1,y2,x1,x2)(bottom,top,left,right)
#             padding_image(image, class_num,100, 100, 100, 100)  #flip_image(image,0)#horizontal
#             # flip_image(image,1, class_num)#vertical
#             # flip_image(image,-1, class_num)#both
#
# create_data(datadir, img_size)


import glob
for dir in os.listdir(data_path):
    directory1 = data_path + dir+'/'
    # print("directory1",directory1)
    # for dir2 in os.listdir(directory1):
    #     directory = directory1+dir2
    #     print("directory",directory)
    try:
        temp_out_path = out_path + str(dir)
        # print("temp_out_putpath", temp_out_path)
        if not os.path.exists(temp_out_path):
            os.makedirs(temp_out_path)
        # for image in os.listdir(directory):
        img_count=0
        temp_out_path = temp_out_path + "/"
        imgs = glob.glob(directory1+"/*.jpg")
        for img_path in glob.glob(directory1+"/*.jpg"):
            print("image count", img_count)
            image = cv2.imread(img_path)


            class_num = imgs.index(img_path)

            padding_image(temp_out_path, image, class_num, 100, 0, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
            padding_image(temp_out_path, image, class_num, 0, 100, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
            padding_image(temp_out_path, image, class_num,0, 0, 100, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
            padding_image(temp_out_path, image, class_num,0, 0, 0, 100)  # (y1,y2,x1,x2)(bottom,top,left,right)
            padding_image(temp_out_path, image, class_num,100, 100, 100, 100)  #flip_image(image,0)#horizontal
            resize_image(image, 450, 400, class_num, temp_out_path)

            # # print("image_path", img_path)
                # g_blur(img_path, temp_out_path,img_count)
                # avg_blur(img_path, temp_out_path,img_count)
                #
                # multiply(img_path,temp_out_path ,img_count)
                # add(img_path, temp_out_path,img_count)
                # filter_color1(img_path, temp_out_path,img_count)
                # filter_color2(img_path, temp_out_path,img_count)
                # filter_color3(img_path, temp_out_path,img_count)
                # shutil.copy(img_path,temp_out_path)


            shutil.copy(img_path, temp_out_path)

            img_count += 1
    except Exception as e:
        print(e.message)
