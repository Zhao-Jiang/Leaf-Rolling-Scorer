#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:58:58 2019

CMD command to use this python script: python LeafRollingScore.py image_patch_path

@author: JiangZhao
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import h5py
import os.path


parser = argparse.ArgumentParser(description='Get the leaf-rolling score from the 160*160 image patch.')
parser.add_argument('img_path', type=str, help='Path to the image patches.')
args = parser.parse_args()
img_path = args.img_path

model = load_model('./LeafRollingScorer_160.h5')
rootdir = img_path
list = os.listdir(rootdir)
num = 0
for i in range(0,len(list)):
    filename = list[i].split('.')[0]
    path = os.path.join(rootdir,list[i])       
    img = image.load_img(path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
    LeafRollingScore = model.predict(x)*5#.reshape([,])
    value = LeafRollingScore[0][0]
    value = ("%.2f"%value)
    print('Leaf-rolling score of image patch '+filename+': '+str(value))
    num=num+1
print(num)