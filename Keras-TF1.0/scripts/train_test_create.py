import shutil
import os
import glob


input_dir  = './alexnet/fake_face_data/dataset'
output_dir = './alexnet/fake_face_data/dataset_splitted'


for classes in os.listdir(input_dir):

    print("Current Folder : ", classes)

    dir_path = os.path.join(input_dir, classes)
    dir_count = glob.glob(dir_path + "/*.jpg")
    count = 0

    for imgs in os.listdir(dir_path):

        img_path = os.path.join(dir_path, imgs)
        count += 1
        train_dir = output_dir + '/' +'train'
        test_dir = output_dir + '/' + 'test'

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        if ((float(count) / float(len(dir_count))) < 0.7):

            out_dir = train_dir + '/'+ classes
            if not os.path.exists(train_dir + '/'+ classes):
                os.mkdir(train_dir + '/'+ classes)
            out_dir = os.path.join(train_dir, classes)
            shutil.copy(img_path, out_dir)

        else:

            out_dir = test_dir + '/' + classes
            if not os.path.exists(test_dir + '/' + classes):
                os.mkdir(test_dir + '/' + classes)
            out_dir = os.path.join(test_dir, classes)
            shutil.copy(img_path, out_dir)

    print('======= Folder Done ======')
