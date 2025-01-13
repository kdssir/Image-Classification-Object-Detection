#!/bin/sh

echo "Hello There..!!!"

echo "Lets Start the Yolo-V4 Training"

########### Update the training parameters here ############## 
############Dont give any space while updateing the parameters######### 

class_list=['diningtable','chair','aeroplane','sofa','person','car','bus','cat','sheep','bottle','tvmonitor','horse','cow','motorbike','bird','train','dog','bicycle','boat','pottedplant']  ### Define your own classes here

## Select Architecture
## 1-Yolo-V4; 2-Tiny-Yolo-V4; 
## 3-Yolo-V3; 4-Tiny-Yolo-V3; 
## 5-Yolo-V2; 6-Tiny-Yolo-V2;

model_arch=4

## Define model size 
task_name=voc_trial
weights_dir=backup_test
batch_size=8
subdivisions=4
learning_rate=0.0001
max_epochs=80000
steps=40000,55000,70000
num_clust=9
m_height=416
m_width=416
gpus=0


if [ "$model_arch" = 5 ] || [ "$model_arch" = 6 ]; then
  v2_flag=1
  num_clust=5
else
  v2_flag=0
fi 

if [ "$model_arch" = 2 ] || [ "$model_arch" = 4 ]; then
  num_clust=6
fi


if [ "$model_arch" = 1 ]; then
  cfg_file="cfg/"$task_name"_yolo-v4.cfg"
  cp /opt/darknet/cfg/yolov4-custom.cfg /opt/darknet/$cfg_file
elif [ "$model_arch" = 2 ]; then
  cfg_file="cfg/"$task_name"_yolo-v4-tiny.cfg"
  cp /opt/darknet/cfg/yolov4-tiny.cfg /opt/darknet/$cfg_file
elif [ "$model_arch" = 3 ]; then
  cfg_file="cfg/"$task_name"_yolo-v3.cfg"
  cp /opt/darknet/cfg/yolov3.cfg /opt/darknet/$cfg_file
elif [ "$model_arch" = 4 ]; then
  cfg_file='cfg/'$task_name'_yolo-v3-tiny.cfg'
  cp /opt/darknet/cfg/yolov3-tiny.cfg /opt/darknet/$cfg_file
elif [ "$model_arch" = 5 ]; then
  cfg_file='cfg/'$task_name'_yolo-v2.cfg'
  cp /opt/darknet/cfg/yolov2.cfg /opt/darknet/$cfg_file
else
  cfg_file='cfg/'$task_name'_yolo-v2-tiny.cfg'
  cp /opt/darknet/cfg/yolov2-tiny.cfg /opt/darknet/$cfg_file
fi


 ## create Data file
cp /opt/darknet/cfg/voc.data /opt/darknet/data/my_data.data

## Copy Del file as 

DIR="/opt/darknet/VOCdevkit/VOC2007/ImageSets"
if [ -d "$DIR" ]; then
  echo "ImageSets is Present"
else
  echo "Creating ImageSets Folder"
  mkdir -p $DIR
fi


DIR="/opt/darknet/VOCdevkit/VOC2007/ImageSets/Main"
if [ -d "$DIR" ]; then
  echo "Main is present inside ImageSets"
else
  echo "Creating Main Folder inside ImageSets"
  mkdir -p $DIR
fi

echo "Derived config name" $cfg_file
echo "v2_flag set as " $v2_flag
echo "number of clusters set as "$num_clust

echo "Started Data Pre-Processing...!!!"

echo "Splitted data in train test"
# python3 /opt/training_scripts/create_file_folders.py  --model_arch $model_arch --task_name $task_name
python3 /opt/training_scripts/chop_utility.py --file_path VOCdevkit/VOC2007/Annotations/ 2>&1 | tee logs/chop_utility.log
cp -r trainfile.txt valfile.txt VOCdevkit/VOC2007/ImageSets/Main/ && 
echo 'label conversion is taking place'
python3 /opt/training_scripts/label_prompting.py  --class_list $class_list 2>&1 | tee logs/label_prompting.log

echo 'generating anchor boxes for the data'
python3 /opt/training_scripts/anchor_creation.py -width $m_width -height $m_height -filelist 2007_trainfile.txt -output_dir anchors/ -num_clusters $num_clust -V2_flag $v2_flag 2>&1 | tee logs/anchor_creation.log

curr_dir=$(echo $PWD)
x=$(cat  $curr_dir/anchors/anchors$num_clust.txt | head -n 1)

ANCHORS=$x

echo 'Updating the necessary files files'

python3 /opt/training_scripts/config_overhaul.py --task_name $task_name --model_arch $model_arch --subdivisions $subdivisions --batch_size $batch_size --anchors $ANCHORS --height $m_height --width $m_width --lr $learning_rate --max_ep $max_epochs --steps $steps --backup $weights_dir --class_list $class_list 2>&1 | tee logs/config_overhaul.log

echo "Starting the training"

################ Update the training command as per customization ############################

./darknet detector train data/my_data.data $cfg_file -map -dont_show -gpus $gpus -mjpeg_port 8090 2>&1 | tee logs/train.log

