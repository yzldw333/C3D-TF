# -*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib.layers import  batch_norm

import json
import os
import cv2
import numpy as np
from scipy.misc import imread,imresize
import random
def weights(shape,name,is_train=True):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.02),trainable=is_train)
        #tf.contrib.layers.xavier_initializer()
def bias(shape,name,is_train=True):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=is_train)

def conv(X,Input_dim,Output_dim,name,stride_size,kernel_size,padding):
    with tf.variable_scope(name) as scope:
        W=weights([kernel_size,kernel_size,Input_dim,Output_dim],'W')
        b=bias([Output_dim],'b')
        return tf.nn.conv2d(X,W,strides=[1,stride_size,stride_size,1],padding=padding)+b




def MaxPooling(X,ksize,stride,padding,name):
    return tf.nn.max_pool(X,ksize=ksize,strides=[1,stride,stride,1],padding=padding,name=name)

def pooling_1x1(Input,input_dim,output_dim,padding,name,stride=2):
    W=weights([1,1,input_dim,output_dim],'pool_w')
    return tf.nn.conv2d(Input,filter=W,strides=[1,stride,stride,1],padding=padding,name=name)

def pooling_3x3(input,name,stride=2,kernel_size=3):
    return tf.nn.max_pool(input,[1,kernel_size,kernel_size,1],strides=[1,stride,stride,1],padding='SAME',name=name)



def BatchNorm(input,train_phase,scope_bn):
    return batch_norm(input,decay=0.99,center=True,scale=True,is_training=train_phase,scope=scope_bn)

def fc(input,input_dim,output_dim,name):
    with tf.variable_scope(name):
        W=weights([input_dim,output_dim],'W')
        b=bias([output_dim],'b')
        return tf.matmul(input,W)+b


def Block(input,conv1_input_dim,conv1_output_dim,stride1_size,kernel1_size,conv_name_1,bn_name_1,
          conv2_input_dim,conv2_output_dim,stride2_size,kernel2_size,conv_name_2,bn_name_2,
          conv3_input_dim,conv3_output_dim,stride3_size,kernel3_size,conv_name_3,bn_name_3,
          pass_way_conv_name,pass_way_bn_name,train_phase,up_dim=False,padding='SAME'):

    conv1=conv(input,conv1_input_dim,conv1_output_dim,conv_name_1,stride1_size,kernel1_size,padding=padding)
    bn_1=BatchNorm(conv1,train_phase,scope_bn=bn_name_1)
    relu_1=tf.nn.relu(bn_1)

    conv_2=conv(relu_1,conv2_input_dim,conv2_output_dim,conv_name_2,stride2_size,kernel2_size,padding=padding)
    bn_2=BatchNorm(conv_2,train_phase,scope_bn=bn_name_2)
    relu_2=tf.nn.relu(bn_2)

    conv_3=conv(relu_2,conv3_input_dim,conv3_output_dim,conv_name_3,stride3_size,kernel3_size,padding=padding)
    bn_3=BatchNorm(conv_3,train_phase,scope_bn=bn_name_3)

    if up_dim==True:
        shape=input.get_shape().as_list()

        conv_3_1=conv(input,shape[3],conv3_output_dim,pass_way_conv_name,stride1_size,1,padding='SAME')
        input=BatchNorm(conv_3_1,train_phase,scope_bn=pass_way_bn_name)

    concat=input+bn_3
    block=tf.nn.relu(concat)

    return block


def data_augmentation_lu(imgs):
    images = []
    size = [((0,0),(0.75,0.75)),((0,0.25),(0.75,1)),((0.25,0),(1,0.75)),((0.25,0.25),(1,1)),
    ((0.15,0.15),(0.9,0.9)),((0,0),(1,1))]
    for data in imgs:
        s = random.choice(size)
        h,w,c = data.shape
        start_h=int(s[0][0]*h)
        start_w=int(s[0][1]*w)
        target_h = int((s[1][0]-s[0][0])*h)
        target_w = int((s[1][1]-s[0][1])*w)

        img = crop_to_bounding_box(data,start_h,start_w,target_h,target_w)
        if np.random.rand()<0.5:
            img = np.flip(img,axis=1)
        img = imresize(img,[224,224])
        images.append(img)
    return np.array(images)

def data_augmentation(imgs,N=256):
    images=[]

    for data in imgs:
        size = [N, 0.875 * N, 0.75 * N, 0.625 * N, 0.5 * N]
        w = int(random.choice(size))
        h=int(random.choice(size))
        choose=np.random.randint(0,2)
        if w==256 and h!=256:
            h_start=N-h
            offset_h=np.random.randint(0,h_start)
            img=crop_to_bounding_box(data,offset_h,0,h,w)
            if choose==1:
                img=np.flip(img,axis=1)
            img=imresize(img,[224,224])
            images.append(img)
        elif h==256 and w!=256:
            w_start=N-w
            offset_w=np.random.randint(0,w_start)
            img=crop_to_bounding_box(data,0,offset_w,h,w)
            if choose==1:
                img=np.flip(img,axis=1)
            img=imresize(img,[224,224])
            images.append(img)

        elif h==256 and w==256:
            img=imresize(data,[224,224])
            if choose==1:
                img=np.flip(img,axis=1)
            images.append(img)
        else:
            w_start=N-w
            h_start=N-h
            offset_w=np.random.randint(0,w_start)
            offset_h=np.random.randint(0,h_start)
            img=crop_to_bounding_box(data,offset_h,offset_w,h,w)
            if choose==1:
                img=np.flip(img,axis=1)
            img=imresize(img,[224,224])
            images.append(img)
    return np.array(images)
    return (np.array(images)-128.0)/255.0
def crop_to_bounding_box(img,offset_h,offset_w,target_h,target_w):
    #原始图片256>=offset+target
    return img[offset_h:offset_h+target_h,offset_w:offset_w+target_w]


def read_image(batch_image,size=256,root=r'E:\dataset\ai_challenger_scene_test_a_20170922\scene_test_a_images_20170922'):
    imgs=[]
    # print('----------------------------------------------------')
    for image in batch_image:
        image= os.path.join(root,os.path.basename(image.strip()))
        im=imread(image)

        # print('start',image)
        if len(im.shape)==2:
            # print('gray---------------------------------')
            im=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        elif im.shape[2]==4:
            im=cv2.cvtColor(im,cv2.COLOR_RGBA2RGB)
        # print('end',im.shape)
        im=imresize(im,(size,size))
        imgs.append(im)
    return imgs
    #return (np.array(imgs)-128.0)/255.0

def one_hot(labels,n):

    arr=np.zeros([len(labels),n])
    for i in range(len(labels)):
        arr[i,int(labels[i].split('\n')[0])]=1.0
    return arr


def load_json(path):
    with open(path,'r') as f:
        data=json.load(f)
        return data

def  read_txt(path):
    with open(path) as f:
        data=f.readlines()
        return data

def orthogonal_initializer(shape,scale=1.0):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    def _initializer(shape, dtype=tf.float32):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer(shape)
if __name__ == '__main__':
    res=orthogonal_initializer(shape=[10,5])
    print('res',res)


