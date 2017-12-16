from utils import *
import tensorflow as tf
#from tensorflow.contrib.keras.api.keras.applications.mobilenet import MobileNet

def mobilenet(X,output_dim=1000,train_phase=True,no_top=False):


    conv1_0=tf.nn.relu(BatchNorm(conv(X,3,32,'conv1',stride_size=2,kernel_size=3,padding='SAME'),train_phase,scope_bn='conv1_bn'))
    conv1=depthwise_pointwise_Conv(conv1_0,32,64,'conv_dw_1','conv_dw_1_bn','conv_pw_1','conv_pw_1_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv2=depthwise_pointwise_Conv(conv1,64,128,'conv_dw_2','conv_dw_2_bn','conv_pw_2','conv_pw_2_bn',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv3=depthwise_pointwise_Conv(conv2,128,128,'conv_dw_3','conv_dw_3_bn','conv_pw_3','conv_pw_3_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv4=depthwise_pointwise_Conv(conv3,128,256,'conv_dw_4','conv_dw_4_bn','conv_pw_4','conv_pw_4_bn',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv5=depthwise_pointwise_Conv(conv4,256,256,'conv_dw_5','conv_dw_5_bn','conv_pw_5','conv_pw_5_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv6=depthwise_pointwise_Conv(conv5,256,512,'conv_dw_6','conv_dw_6_bn','conv_pw_6','conv_pw_6_bn',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv7=depthwise_pointwise_Conv(conv6,512,512,'conv_dw_7','conv_dw_7_bn','conv_pw_7','conv_pw_7_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv8=depthwise_pointwise_Conv(conv7,512,512,'conv_dw_8','conv_dw_8_bn','conv_pw_8','conv_pw_8_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv9=depthwise_pointwise_Conv(conv8,512,512,'conv_dw_9','conv_dw_9_bn','conv_pw_9','conv_pw_9_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv10=depthwise_pointwise_Conv(conv9,512,512,'conv_dw_10','conv_dw_10_bn','conv_pw_10','conv_pw_10_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv11=depthwise_pointwise_Conv(conv10,512,512,'conv_dw_11','conv_dw_11_bn','conv_pw_11','conv_pw_11_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)


    conv12=depthwise_pointwise_Conv(conv11,512,1024,'conv_dw_12','conv_dw_12_bn','conv_pw_12','conv_pw_12_bn',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv13=depthwise_pointwise_Conv(conv12,1024,1024,'conv_dw_13','conv_dw_13_bn','conv_pw_13','conv_pw_13_bn',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    avgPooling=AvgPooling(conv13,[1,7,7,1],1,'VALID','avgpooling')
    endpoints=[conv1_0,conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11,conv12,conv13,avgPooling]
    if no_top==False:
        features = tf.reshape(avgPooling,[-1,1024])
        logits=fc(features,1024,output_dim,'fc',wd=5e-4)
    endpoints.append(logits)
    return endpoints




def load_pretrained_mobilenet_model_ops(model='mobilenet_1_0_224_tf.h5',batchsize=24):
    import h5py
    file = h5py.File(model)
    dct={}
    vars = tf.global_variables()
    for e in vars:
        dct[e.name]=e
    print('******************************************************')
    ops=[]
    def search_and_assign(node):
        for key,value in node.items():
            if 'bn' in key:
                for key,value in value.items():
                    for subkey,value in value.items():
                        if subkey=='gamma:0':
                            joinkey=key+'/'+subkey
                            if joinkey in dct:
                                ops.append(tf.assign(dct[joinkey],value.value))
                        elif subkey=='beta:0':
                            joinkey=key+'/'+subkey
                            if joinkey in dct:
                                ops.append(tf.assign(dct[joinkey],value.value))
                        elif subkey=='moving_mean:0':
                            joinkey=key+'/'+subkey
                            if joinkey in dct:
                                ops.append(tf.assign(dct[joinkey],value.value))
                        elif subkey=='moving_variance:0':
                            joinkey=key+'/'+subkey
                            if joinkey in dct:
                                ops.append(tf.assign(dct[joinkey],value.value))
            elif 'conv'or 'fc' in key:
                for key,value in value.items():
                    for subkey,value in value.items():
                        print(subkey)
                        joinkey=''
                        if subkey == 'kernel:0':
                            subkey='W:0'
                            joinkey=key+'/'+subkey
                        elif subkey=='depthwise_kernel:0':
                            joinkey=key+'/'+subkey
                        if joinkey not in dct:
                            continue
                        ops.append(tf.assign(dct[joinkey],value.value))
    search_and_assign(file)
    return ops
