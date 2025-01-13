import os
from os import  getcwd
wd = getcwd()
txt_fle = open(wd+'/somefile.txt', 'r')


for aline in txt_fle:
	print(aline)
	x = aline.split('/')[1]
	print(x)
	os.remove(os.path.join(wd + '/Annotations', x.split('.txt')[0] + '.xml'))
	os.remove(os.path.join(wd + '/JPEGImages', x.split('.txt')[0] + '.jpg'))
