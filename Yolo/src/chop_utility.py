import os
import argparse


writer = open("valfile.txt", "w+")
writer2 = open("trainfile.txt", "w+")


def parse_args():
    parser = argparse.ArgumentParser(description='Input File Path')
    parser.add_argument('--file_path', dest='file_path', help='create train txt',
                        default='pascal', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
files = os.listdir(args.file_path)
print ("No  of files", len(files))
total_files = len(files)
val_count = int(total_files * 0.2)
val_files = []
train_count = total_files - val_count
print ("Total train and val files", train_count, val_count)
for root, directory, files in os.walk(args.file_path):
    for file in files:
        if val_count != 0:
            writer.write(str(file.split(".xml")[0]))
            writer.write("\n")
            val_files.append(str(file.split(".xml")[0]))
            val_count -= 1
    # print "File ", file
    else:
        break

for root, directory, files in os.walk(args.file_path):
    for file in files:
        if str(file.split(".xml")[0]) not in val_files:
            writer2.write(str(file.split(".xml")[0]))
            writer2.write("\n")
            # val_files.append(str(file.split(".")[0]))
            # print "File ", file

writer.close()
writer2.close()
