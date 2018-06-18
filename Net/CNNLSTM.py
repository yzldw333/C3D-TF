import tensorflow as tf
from mobilenet import *
from resnet50 import *
from utils import *
# classes
NUM_CLASSES = 19

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
WIDTH = 224
HEIGHT = 224
CHANNELS = 3



def focal_loss(onehot_labels,logits,alpha=0.25,gamma=2.0,name=None,scope=None):
    """
    logits and onehot_labels must have same shape[batchsize,num_classes] and the same data type(float16 32 64)
    Args:
        onehot_labels: [batchsize,classes]
        logits: Unscaled log probabilities(tensor)
        alpha: The hyperparameter for adjusting biased samples, default is 0.25
        gamma: The hyperparameter for penalizing the easy labeled samples
        name: A name for the operation(optional)

    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    precise_logits = tf.cast(logits,tf.float32) if (
            logits.dtype==tf.float16) else logits
    onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
    predictions = tf.nn.softmax(precise_logits)
    predictions_pt = tf.where(tf.equal(onehot_labels,1),predictions,1.-predictions)
    epsilon = 1e-8
    alpha_t = tf.scalar_mul(alpha,tf.ones_like(onehot_labels,dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels,1.0),alpha_t,1-alpha_t)
    losses = tf.reduce_mean(-alpha_t*tf.pow(1.-predictions_pt,gamma)*onehot_labels*tf.log(predictions_pt+epsilon),
            name=name)
    #tf.summary.scalar(
    #    scope + '-focal_loss',
    #    losses
    #)
    return losses

def lstm(x,hidden_size,batchsize):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.7)
    #lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LSTM_LAYERS , state_is_tuple=True)
    init_state = lstm_cell.zero_state(batchsize,dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, time_major=False,initial_state=init_state)
    return outputs

def inference_mobilenet_lstm(batchsize,time_steps=4,hidden_size=50,classes=19,loss='focal_loss',train_phase=True):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")

    endpoints = mobilenet(X,output_dim=hidden_size,train_phase=train_phase,no_top=False)
    features = endpoints[-1]
    features = tf.reshape(features,[-1,time_steps,hidden_size])
    endpoints.append(features)
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    #output = output[:,-1]
    #logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    output = tf.reshape(output,[-1,time_steps*hidden_size])
    logits = fc(output,input_dim=hidden_size*time_steps,output_dim=classes,name="fc_logits")
    if loss=='focal_loss':
        import tensorflow.contrib.slim as slim
        one_hot_labels = slim.one_hot_encoding(Y, classes)
        mean_loss = focal_loss(one_hot_labels,logits,alpha=0.25,gamma=2.0,name='focal_loss')
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
        mean_loss = tf.reduce_mean(loss)

    predict = tf.nn.softmax(logits,name="predict")
    predict = tf.argmax(predict, 1)

    correct_pred = tf.equal(predict, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,endpoints,features,predict,mean_loss,accuracy

def inference_attention_mobilenet_lstm(batchsize,time_steps=4,hidden_size=50,classes=19,loss='focal_loss',train_phase=True):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")

    endpoints = mobilenet(X,output_dim=hidden_size,train_phase=train_phase,no_top=False)
    features = endpoints[-1]
    features = tf.reshape(features,[-1,time_steps,hidden_size])
    endpoints.append(features)
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    #output = output[:,-1]
    #logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    output = tf.reshape(output,[-1,time_steps*hidden_size])
    attention = tf.nn.sigmoid(fc(output,input_dim=hidden_size*time_steps,output_dim=hidden_size*time_steps,name='att'))
    output =  tf.multiply(output,attention)

    logits = fc(output,input_dim=hidden_size*time_steps,output_dim=classes,name="fc_logits")
    if loss=='focal_loss':
        import tensorflow.contrib.slim as slim
        one_hot_labels = slim.one_hot_encoding(Y, classes)
        mean_loss = focal_loss(one_hot_labels,logits,alpha=0.25,gamma=2.0,name='focal_loss')
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
        mean_loss = tf.reduce_mean(loss)

    predict = tf.nn.softmax(logits,name="predict")
    predict = tf.argmax(predict, 1)

    correct_pred = tf.equal(predict, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,endpoints,features,predict,mean_loss,accuracy

def inference_channel_attention_mobilenet_lstm(batchsize,time_steps=4,hidden_size=50,classes=19,loss='focal_loss',train_phase=True):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")

    endpoints = mobilenet(X,output_dim=hidden_size,train_phase=train_phase,no_top=False)
    features = endpoints[-1]
    features = tf.reshape(features,[-1,time_steps,hidden_size])
    endpoints.append(features)
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    #output = output[:,-1]
    #logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    output = tf.reshape(output,[-1,time_steps,hidden_size,1])
    output_trans = tf.transpose(output,[0,2,3,1])
    avg = tf.nn.avg_pool(output_trans,ksize=[1,hidden_size,1,1],strides=[1,hidden_size,1,1],padding='VALID',name='channelattAvg')
    attconv1 = conv(avg,time_steps,time_steps,name='attconv1',kernel_size=1,stride_size=1,padding='VALID',wd=0)
    attconv2 = conv(attconv1,time_steps,time_steps,name='attconv2',kernel_size=1,stride_size=1,padding='VALID',wd=0)
    attention = tf.nn.softmax(attconv2)
    attention = tf.reshape(attention,[-1,time_steps,1,1])

    output = tf.multiply(output,attention)
    output = tf.reshape(output,[-1,time_steps*hidden_size])

    logits = fc(output,input_dim=hidden_size*time_steps,output_dim=classes,name="fc_logits")
    if loss=='focal_loss':
        import tensorflow.contrib.slim as slim
        one_hot_labels = slim.one_hot_encoding(Y, classes)
        mean_loss = focal_loss(one_hot_labels,logits,alpha=0.25,gamma=2.0,name='focal_loss')
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
        mean_loss = tf.reduce_mean(loss)

    predict = tf.nn.softmax(logits,name="predict")
    predict = tf.argmax(predict, 1)

    correct_pred = tf.equal(predict, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,endpoints,features,predict,mean_loss,accuracy

def inference_resnet_lstm(batchsize,time_steps=4,hidden_size=50,classes=19,loss='focal_loss',train_phase=True):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")
    features,block4 = resnet50_BVLC(X,output_dim=hidden_size,no_top=False,train_phase=train_phase)
    features = tf.reshape(features,[-1,time_steps,hidden_size])
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    #output = output[:,-1]
    #logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    output = tf.reshape(output,[-1,time_steps*hidden_size])
    logits = fc(output,input_dim=hidden_size*time_steps,output_dim=classes,name="fc_logits")
    predict = tf.nn.softmax(logits,name="predict")
    predict = tf.argmax(predict, 1)
    if train_phase==False:
        return X,predict 
    if loss=='focal_loss':
        import tensorflow.contrib.slim as slim
        one_hot_labels = slim.one_hot_encoding(Y, classes)
        mean_loss = focal_loss(one_hot_labels,logits,alpha=0.25,gamma=2.0,name='focal_loss')
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
        mean_loss = tf.reduce_mean(loss)

    correct_pred = tf.equal(predict, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,predict,mean_loss,accuracy


def inference_c3d_lstm(batchsize,time_steps=4,hidden_size=50,classes=19,loss='focal_loss',train_phase=True):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")
    features,block4 = resnet50_BVLC(X,output_dim=hidden_size,no_top=False,train_phase=train_phase)
    features = tf.reshape(features,[-1,time_steps,hidden_size])
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    #output = output[:,-1]
    #logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    output = tf.reshape(output,[-1,time_steps*hidden_size])
    logits = fc(output,input_dim=hidden_size*time_steps,output_dim=classes,name="fc_logits")
    if loss=='focal_loss':
        import tensorflow.contrib.slim as slim
        one_hot_labels = slim.one_hot_encoding(Y, classes)
        mean_loss = focal_loss(one_hot_labels,logits,alpha=0.25,gamma=2.0,name='focal_loss')
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
        mean_loss = tf.reduce_mean(loss)
    predict = tf.nn.softmax(logits,name="predict")
    predict = tf.argmax(predict, 1)

    correct_pred = tf.equal(predict, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,predict,mean_loss,accuracy
