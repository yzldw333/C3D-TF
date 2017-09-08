# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
def get_frames_data(filename, num_frames_per_clip=16,temporal_elastic_deformation=False,random_dropping=False,random_rotate_range=0,random_scale_range=0,random_shift_range=(0,0)):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    if temporal_elastic_deformation==True:
      clf = Pipeline([('poly', PolynomialFeatures(degree=2)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
      originLength = len(filenames)
      M = originLength 
      n = random.normalvariate(mu=M,sigma=3)
      m = random.normalvariate(mu=n,sigma=4*(1-abs(n-M)/M))
      x = np.array([0,n,M-1])
      y = np.array([0,m,M-1])
      clf.fit(x[:, np.newaxis], y)  
      x_test = np.arange(0,len(filenames),1)
      y_test = clf.predict(x_test[:, np.newaxis])
      y_test = [int(e) for e in y_test]
      ss_index = []
      for i in range(num_frames_per_clip):
        index = int(1.0*i*originLength/num_frames_per_clip)
        if index>=originLength:
          index = originLength-1
        value=y_test[index]
        if value>=originLength:
          value = originLength-1
        if value<0:
          value = 0
        ss_index.append(value)
      s_index = ss_index
      #import matplotlib.pyplot as plt
      #plt.plot(x_test,y_test,linewidth=2)
      #plt.show()
    else:
      #start_index = random.randint(0, len(filenames) - num_frames_per_clip+1)
      #s_index = [start_index+i for i in range(num_frames_per_clip)]
      # average index
      s_index = []
      originLength = len(filenames)
      for i in range(num_frames_per_clip):
        index = int(i/num_frames_per_clip*originLength)
        if index>originLength-1:
          index=originLength-1
        s_index.append(index)

    rotate_angle = 0
    scale = 1
    shift_x = 0
    shift_y = 0
    shift_matrix = np.float32([[1,0,0],[0,1,0]])
    rotate_matrix = np.float32([[1,0,0],[0,1,0]])

    for i in range(num_frames_per_clip):
      index = s_index[i]
      image_name = str(filename) + '/' + str(filenames[index])
      img = Image.open(image_name)
      img_data = np.array(img)
      (h, w) = img_data.shape[:2]
      center = (w / 2, h / 2)
      if i==0:
        # construct affine matrix
        if random_rotate_range>0:
          rotate_angle = np.random.uniform(-random_rotate_range,random_rotate_range)
        if random_scale_range>0:
          scale = np.random.uniform(1-random_scale_range,1+random_scale_range)
        if random_shift_range[0]>0:
          shift_x = np.random.uniform(-random_shift_range[0],random_shift_range[0])
        if random_shift_range[1]>0:
          shift_y = np.random.uniform(-random_shift_range[1],random_shift_range[1])
        shift_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
        rotate_matrix = cv2.getRotationMatrix2D(center, rotate_angle, scale)
      # shift and rotate
      shifted = cv2.warpAffine(img_data,shift_matrix,(w,h))
      img_data = cv2.warpAffine(shifted, rotate_matrix, (w, h))
      # random_dropping
      if random_dropping == True:
        random_dropping_arr = np.random.rand(img_data.shape[0],img_data.shape[1],img_data.shape[2])
        img_data[random_dropping_arr<0.3]=0
      ret_arr.append(img_data)
  return ret_arr, s_index

def read_clip_and_label(rootdir,filename,batch_size, lines=None,start_pos=-1, num_frames_per_clip=16, crop_size=(112,112), shuffle=False,phase='TRAIN'):
  if lines==None:
    lines = open(filename,'r')
    lines = list(lines)
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  #np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size[0], crop_size[1], 3])
  np_mean = np.ones([num_frames_per_clip,crop_size[0],crop_size[1],3])*128

  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
    start_pos=0
  if shuffle:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(lines)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for i in range(len(video_indices)+1):
    if i >= len(video_indices):
      next_batch_start = -1
      break
    index = video_indices[i]
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    if os.path.exists(rootdir) == True:
      dirname = os.path.join(rootdir,dirname)

    tmp_label = line[2] # to serve C3D caffe project
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    if phase == 'TRAIN':
      tmp_data, _ = get_frames_data(dirname, num_frames_per_clip,
                                      temporal_elastic_deformation=False,
                                      random_dropping=False)
                                      #random_rotate_range=10,
                                      #random_scale_range=0.3) # default open temporal elastic deformation
    elif phase == 'TEST':
      tmp_data, _ = get_frames_data(dirname, num_frames_per_clip,temporal_elastic_deformation=False,random_dropping=False)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in range(len(tmp_data)):
        img = tmp_data[j].astype(np.uint8)
        if img.shape[0]!=crop_size[0] or img.shape[1]!=crop_size[1]:
          img = np.array(cv2.resize(img,(crop_size[1],crop_size[0]))).astype(np.float32)
        else:
          img = np.array(img).astype(np.float32)
        img-=128
        img/=128.0
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  print('next_batch_start:%d'%next_batch_start)
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len,lines


if __name__ == '__main__':
  tmp_data, _ = get_frames_data(r'E:\dataset\VIVA_avi_group\VIVA_avi_part0\train\03_01_01', 16,temporal_elastic_deformation=True) 
  for e in tmp_data:
    import cv2
    cv2.imshow('t',e)
    cv2.waitKey(0)

  pass