#!/bin/sh

echo "Hello There..!!!"

echo "Lets Starte the Training"

## Define model size
data_path=/opt/imagenet/data/mask/mask_augmented_dataset
learning_rate=0.001
max_epochs=300
num_classes=2
gpus=1
batch_size=256
model_name=ResNet18_mask_augmented_inclusive

echo "--------------------- Training is getting started ------------------------------"
CUDA_VISIBLE_DEVICES=$gpus python3 main_112.py -b $batch_size --lr $learning_rate --epochs $max_epochs --classes $num_classes --model_name $model_name --data $data_path  | tee train.log 


echo "--------------------- Onnx Conversion is getting done ------------------------------"
python3 convert_to_onnx.py --model_name $model_name --classes $num_classes
