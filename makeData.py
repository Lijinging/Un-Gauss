# coding:utf-8

import os

import numpy
import scipy.io as sio
from PIL import Image

gauss_path = r'IMG_GAUSS/'
raw_path = r'IMG_RAW/'
mat_path = r'data/'
testNum = 20

# 以下创建mat文件
train_label = numpy.zeros((0, 61009), dtype="int")
train_data = numpy.zeros((0, 61009), dtype="int")
test_label = numpy.zeros((0, 61009), dtype="int")
test_data = numpy.zeros((0, 61009), dtype="int")

filelist = os.listdir(raw_path)
numpy.random.shuffle(filelist)
print(len(filelist))
cnt = 0

for infile in filelist:

    cnt = cnt + 1
    print(cnt, infile)

    if cnt >= testNum:
        img_train_data = numpy.array(Image.open(raw_path + infile).convert("L").resize((247, 247), Image.ANTIALIAS)).reshape(1,
                                                                                                                61009)
        train_data = numpy.row_stack((train_data, img_train_data))
        img_train_label = numpy.array(Image.open(gauss_path + infile).convert("L").resize((247, 247), Image.ANTIALIAS)).reshape(1,
                                                                                                                   61009)
        train_label = numpy.row_stack((train_label, img_train_label))
    else:
        img_test_data = numpy.array(Image.open(raw_path + infile).convert("L").resize((247, 247), Image.ANTIALIAS)).reshape(1, 61009)
        test_data = numpy.row_stack((test_data, img_test_data))
        img_test_label = numpy.array(Image.open(gauss_path + infile).convert("L").resize((247, 247), Image.ANTIALIAS)).reshape(1,
                                                                                                                  61009)
        test_label = numpy.row_stack((test_label, img_test_label))

sio.savemat(mat_path + 'train.mat', {'data': train_data, 'label': train_label})
sio.savemat(mat_path + 'test.mat', {'data': test_data, 'label': test_label})
