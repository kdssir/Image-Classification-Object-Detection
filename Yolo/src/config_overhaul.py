import os
import argparse

data_file = './data/my_data.data'

names_file  = './data/obj.names'
 
 

def save_trainVal_configs(config_file, args):

    config_name = config_file.split('.cfg')[0].split('/')[2]

    train_cfg_file = "./exported_model/"+config_name + '_train.cfg'
    test_cfg_file = "./exported_model/"+config_name + '_test.cfg'
    os.system("cp -r "+config_file + "  " + train_cfg_file)
    os.system("cp -r "+config_file + "  " + test_cfg_file)

    with open(train_cfg_file, 'r') as file:
    # read a list of lines into data
        data = file.readlines()
    
    data [2] = '#batch= 1' + '\n'
    data [3] = '#subdivisions= 1' + '\n' 
    data [5] = 'batch='+args.batch_size +'\n'
    data [6] =  'subdivisions= ' + args.subdivisions + '\n'

    with open(train_cfg_file, 'w') as file:
        file.writelines( data )

    data = []

    with open(test_cfg_file, 'r') as file:
    # read a list of lines into data
        data = file.readlines()

    data [2] = 'batch= 1' + '\n'
    data [3] = 'subdivisions= 1' + '\n' 
    data [5] = '#batch= 8'+'\n'
    data [6] =  '#subdivisions= 4' + '\n'

    with open(test_cfg_file , 'w') as file:
        file.writelines( data )
    

def parse_args():
    parser = argparse.ArgumentParser(description='Input File Path')
    
    #### For config file ######
    parser.add_argument('--anchors', dest='anchors', help='anchors',
                        default='pascal', type=str)
    
    parser.add_argument('--height', dest='height', help='image height',
                        default='pascal', type=str)
    
    parser.add_argument('--width', dest='width', help='image width',
                        default='pascal', type=str)
    
    parser.add_argument('--lr', dest='lr', help='Learning Rate',
                        default='pascal', type=str)
    
    parser.add_argument('--max_ep', dest='max_ep', help='max epochs',
                        default='pascal', type=str)
    
    parser.add_argument('--steps', dest='steps', help='steps for LR reduction',
                        default='pascal', type=str)
    
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default='pascal', type=str)

    parser.add_argument('--subdivisions', dest='subdivisions', help='batch size',
                        default='pascal', type=str)

    parser.add_argument('--model_arch', dest='model_arch', help='model architecture',
                       default='1', type=int)

    parser.add_argument('--task_name', dest='task_name', help='task name',
                       default='training', type=str)
    
    ### for data file ######
    parser.add_argument('--backup', dest='backup', help='create train txt',
                        default='pascal', type=str)
    
    ### for names file #####
    
    parser.add_argument('--class_list', dest='class_list', help='create train txt',
                        default='pascal', type=str)
    
    args = parser.parse_args()
    return args

args = parse_args()

name_dir = {1:"yolo-v4.cfg",2:'yolo-v4-tiny.cfg',3:'yolo-v3.cfg', 4:'yolo-v3-tiny.cfg', 5:'yolo-v2.cfg', 6:'yolo-v2-tiny.cfg'}

steps = args.steps.split(',')

cfg_file = './cfg/'  +args.task_name+'_'+name_dir[args.model_arch]

scls = ''
for i in range(len(steps)):
    if not i == len(steps) - 1:
        scls += '.1,' 
    else:
        scls += '.1'
        
classes = []
z = args.class_list.split(',')
for i, item in enumerate(z):
    if i == 0:
        item = item.split('[')[1]
    item = item.split(']')[0]
    classes.append(item)

anchrs = args.anchors.split(',')
num_classes = len(classes)
new_anchrs = ''

for i, item in enumerate(anchrs):
    if not i == len(anchrs) -1:
        if not (i % 2) == 0:
            new_anchrs += item +',' +' '
        else:
            new_anchrs += item + ','
    else:
        new_anchrs += item
 


###### Updating the config file ########

########################## YOLO-V4 ##########################

print('[INFO]... Updating the config file')

arch = name_dir[args.model_arch].split('.cfg')[0]

print('[INFO]... Training will start for ', arch)
with open(cfg_file, 'r') as file:
    # read a list of lines into data
    data = file.readlines()
    
print('\t Updataing model size as per specified')
data [2] = '#batch = 1' + '\n'
data [3] = '#subdivisions = 1' + '\n' 
data [5] = 'batch='+args.batch_size +'\n'
data [6] =  'subdivisions= ' + args.subdivisions + '\n'
data [7] = 'width=' + args.width + '\n'
data [8] = 'height='+args.height +'\n'

print('\t Updataing Learning Rate as per specified')
data [17] = 'learning_rate=' + args.lr +'\n'
print('\t Updataing Maxx Epochs as per specified')
data [19] = 'max_batches = ' + args.max_ep+'\n'
print('\t Updataing Steps as per specified')
data [21] = 'steps=' + args.steps+'\n'
print('\t Updataing Sclaes per steps as per specified')
data [22] = 'scales=' +  scls


if args.model_arch == 1:
    print('\t Updataing the Yolo Layer 1 as per ', arch, ' Model Architecture')
    data [962] = 'filters=' + str((num_classes+5)*3) +'\n'
    data [968] = 'anchors= ' + new_anchrs + '\n'
    data [969] = 'classes= ' +  str(num_classes)+ '\n'

    print('\t Updataing the Yolo Layer 2 as per ',arch, ' Model Architecture ')
    data [1050] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [1056] = 'anchors= ' + new_anchrs+'\n' 
    data [1057] = 'classes= ' +  str(num_classes)+'\n'

    print('\t Updataing the Yolo Layer 3 as per ',arch, ' Model Architecture')
    data [1138] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [1144] = 'anchors= ' + new_anchrs +'\n'
    data [1145] = 'classes= ' +  str(num_classes)+'\n'

elif args.model_arch == 2:
    print('\t Updataing the Yolo Layer 1 as per ',arch, ' Model Architecture')
    data [211] = 'filters=' + str((num_classes+5)*3) +'\n'
    data [218] = 'anchors= ' + new_anchrs + '\n'
    data [219] = 'classes= ' +  str(num_classes)+ '\n'

    print('\t Updataing the Yolo Layer 2 as per ',arch, ' Model Architecture')
    data [262] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [267] = 'anchors= ' + new_anchrs+'\n' 
    data [268] = 'classes= ' +  str(num_classes)+'\n'

elif args.model_arch == 3:

    print('\t Updataing the Yolo Layer 1 as per ',arch, ' Model Architecture')
    data [602] = 'filters=' + str((num_classes+5)*3) +'\n'
    data [608] = 'anchors= ' + new_anchrs + '\n'
    data [609] = 'classes= ' +  str(num_classes)+ '\n'

    print('\t Updataing the Yolo Layer 2 as per ',arch, ' Model Architecture')
    data [688] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [694] = 'anchors= ' + new_anchrs+'\n' 
    data [695] = 'classes= ' +  str(num_classes)+'\n'

    print('\t Updataing the Yolo Layer 3 as per ',arch, ' Model Architecture')
    data [775] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [781] = 'anchors= ' + new_anchrs +'\n'
    data [782] = 'classes= ' +  str(num_classes)+'\n'

elif  args.model_arch == 4:

    print('\t Updataing the Yolo Layer 1 as per ',arch, ' Model Architecture')
    data [126] = 'filters=' + str((num_classes+5)*3) +'\n'
    data [133] = 'anchors= ' + new_anchrs + '\n'
    data [134] = 'classes= ' +  str(num_classes)+ '\n'

    print('\t Updataing the Yolo Layer 2 as per ',arch, ' Model Architecture')
    data [170] = 'filters=' + str((num_classes+5)*3)+'\n'
    data [175] = 'anchors= ' + new_anchrs+'\n' 
    data [176] = 'classes= ' +  str(num_classes)+'\n'

elif args.model_arch == 5:

    print('\t Updataing the Yolo Layer 1 as per ',arch, ' Model Architecture')
    data [236] = 'filters=' + str((num_classes+5)*5) +'\n'
    data [241] = 'anchors= ' + new_anchrs + '\n'
    data [243] = 'classes= ' +  str(num_classes)+ '\n'

else:
    print('\t Updataing the Yolo Layer 1 as per ' ,arch, ' Model Architecture')
    data [118] = 'filters=' + str((num_classes+5)*5) +'\n'
    data [122] = 'anchors= ' + new_anchrs + '\n'
    data [124] = 'classes= ' +  str(num_classes)+ '\n'


with open(cfg_file, 'w') as file:
    file.writelines( data )

save_trainVal_configs(cfg_file, args)


    


with open(data_file, 'r') as file:
    data = file.readlines()
    
print('[INFO]... Updating the data file')
data[0] = 'classes= ' + str(num_classes) +'\n'
data[1] = 'train = '+ os.getcwd() + '/2007_trainfile.txt\n'
data[2] = 'valid = '+ os.getcwd() + '/2007_valfile.txt\n'
data[3] = 'names = ' + os.getcwd() + '/data/obj.names\n'
data[4] = 'backup = ' + args.backup+'\n'

if not os.path.exists(os.getcwd() + '/' + args.backup):
    os.mkdir(os.getcwd() + '/' + args.backup)



with open(data_file, 'w') as file:
    file.writelines( data )
print('[INFO]... Updating the class list file')
with open(names_file, 'w') as file:
    for item in classes:
        file.write(item+'\n')
print('[INFO]... Completed Updating all the necessary files')

os.system("cp -r  "+data_file + "  " + names_file + "  ./exported_model")
