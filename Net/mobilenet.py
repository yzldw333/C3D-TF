import tensorflow as tf
import keras.applications.mobilenet
model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
def mobilenet(X,output_dim=1000,train_phase=True,no_top=False):

    conv0=tf.nn.relu(BatchNorm(conv(X,3,32,'conv0',stride_size=2,kernel_size=3,padding='SAME'),train_phase,scope_bn='bn_conv0'))
    conv1=depthwise_pointwise_Conv(conv0,32,64,'conv1',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv2=depthwise_pointwise_Conv(conv1,64,128,'conv2',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv3=depthwise_pointwise_Conv(conv2,128,128,'conv3',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv4=depthwise_pointwise_Conv(conv3,128,256,'conv4',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv5=depthwise_pointwise_Conv(conv4,256,256,'conv5',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv6=depthwise_pointwise_Conv(conv5,256,512,'conv6',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv7=depthwise_pointwise_Conv(conv6,512,512,'conv7',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv8=depthwise_pointwise_Conv(conv7,512,512,'conv8',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv9=depthwise_pointwise_Conv(conv8,512,512,'conv9',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv10=depthwise_pointwise_Conv(conv9,512,512,'conv10',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)
    conv11=depthwise_pointwise_Conv(conv10,512,512,'conv11',stride_size=1,kernel_size=3,padding='SAME',train_phase=train_phase)


    conv12=depthwise_pointwise_Conv(conv11,512,1024,'conv12',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)

    conv13=depthwise_pointwise_Conv(conv12,1024,1024,'conv13',stride_size=2,kernel_size=3,padding='SAME',train_phase=train_phase)
    avgPooling=AvgPooling(conv13,7,1,'SAME','avgpooling')
    if no_top==True:
        return avgPooling
    else:
        fc=fc(avgPooling,1024,output_dim,'fc')
        return fc

def load_pretrained_model_ops(model='mobilenet_tf_dim_ordering_tf_kernels.h5'):
    import h5py
    file = h5py.File(model)
    vars = tf.trainable_variables()
    #print(vars)
    dct={}
    for e in vars:
        dct[e.name]=e
    ops = []
    for key,value in file.items():
        if 'bn' in key:
            for key,value in value.items():
                key = key[:-2]
                if key.endswith('gamma'):
                    key = key[:-6]+'/gamma:0'
                    ops.append(tf.assign(dct[key],value.value))
                elif key.endswith('beta'):
                    key = key[:-5]+'/beta:0'
                    ops.append(tf.assign(dct[key], value.value))
                #print(key,value)
        elif 'conv'or 'fc' in key:
            for key,value in value.items():
                key = key[:-4]+'/'+key[-3:]
                if key not in dct:
                    continue
                ops.append(tf.assign(dct[key],value.value))
                #print(key,value)
        elif 'branch' in key:
            for key,value in value.items():
                key = key[:-4] + '/' + key[-3:]
                ops.append(tf.assign(dct[key], value.value))
                #print(key,value)
    return ops

