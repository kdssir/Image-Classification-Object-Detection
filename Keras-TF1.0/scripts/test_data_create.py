import os, shutil
wd = os.getcwd()

for dirs in os.listdir(wd+ '/data_new_org'):
	dir_path = os.path.join(wd+ '/data_new_org', dirs)
	if not os.path.exists(wd+ "/data_test/"+dirs):
		os.mkdir(wd + "/data_test/"+dirs)
		print('dir created')
	for img in os.listdir(dir_path):
		img_path =  os.path.join(dir_path, img)
		if not os.path.exists(os.path.join(wd + '/data_train/'+dirs, img)):
			shutil.copy( img_path , wd + "/data_test/"+dirs)
