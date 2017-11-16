import tensorflow as tf
from resnet50 import *
from utils import *
# classes
NUM_CLASSES = 19

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
WIDTH = 224
HEIGHT = 224
CHANNELS = 3


# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 4

def lstm(x,hidden_size,batchsize):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batchsize,dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, time_major=False,initial_state=init_state)
    #outputs = tf.reshape([-1,NUM_FRAMES_PER_CLIP,hidden_size])
    return outputs[:,-1]

def inference_resnet_lstm(batchsize,time_steps=4,hidden_size=50,classes=19):
    # in X, batchsize = origin_batchsize*time_step
    X = tf.placeholder(dtype=tf.float32,shape=[None,HEIGHT,WIDTH,CHANNELS],name="input")
    Y = tf.placeholder(dtype=tf.int64,shape=[None],name="label")
    #lr = tf.placeholder(dtype=tf.float32,name="learning_rate")
    features,block4 = resnet50_BVLC(X,output_dim=50,no_top=False,train_phase=True)
    features = tf.reshape(features,[-1,time_steps,50])
    print(features.shape)
    # last node's output
    output = lstm(x=features,hidden_size=hidden_size,batchsize=batchsize)
    logits = fc(output,input_dim=hidden_size,output_dim=classes,name="fc_logits")
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='hard_loss')
    mean_loss = tf.reduce_mean(loss)
    predict = tf.nn.softmax(logits,name="predict")

    correct_pred = tf.equal(tf.argmax(predict, 1), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return X,Y,predict,mean_loss,accuracy


