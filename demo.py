import cv2
import threading
import time
import tensorflow as tf
import sys
sys.path.append('Net/')
from Net import C3DModel
from Net import CNNLSTM
import numpy as np
_videoBuffer = []
_gestureId = -1
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return var

def VideoRealtimeProcess(videoBuffer=[],frameNum=16):
    width = 171
    height= 128
    width=224
    height=224
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret==False:
            return
        framecopy = frame.copy()

        grayimg = cv2.cvtColor(framecopy, cv2.COLOR_BGR2GRAY)
        grayimg = cv2.resize(grayimg, (width, height), interpolation=cv2.INTER_AREA)
        grad1 = cv2.Sobel(grayimg, cv2.CV_64F, 1, 0)
        grad1 = cv2.convertScaleAbs(grad1)
        grad2 = cv2.Sobel(grayimg, cv2.CV_64F, 0, 1)
        grad2 = cv2.convertScaleAbs(grad2)
        grad = cv2.addWeighted(grad1, 0.5, grad2, 0.5, 0)
        input_arr = np.ones(shape=(3, height, width), dtype=np.uint8)
        input_arr[0, :, :] = grayimg.reshape([1, height, width])
        input_arr[1, :, :] = grad.reshape([1, height, width])
        input_arr[2, :, :] = input_arr[0,:,:]
        input_arr = input_arr.transpose([1,2,0])
        if cv2.waitKey(1) and 0xFF==ord('q'):
            return
        if len(videoBuffer)<frameNum:
            videoBuffer.append(input_arr)
            continue
        else:
            videoBuffer.pop(0)
            videoBuffer.append(input_arr)


def WindowsRealtimeShow():
    global _gestureId,_videoBuffer
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        if cv2.waitKey(1) and 0xFF==ord('q'):
            return
        if len(_videoBuffer)!=0:
            img = _videoBuffer[-1]
            drawimg = img.copy()
            cv2.putText(drawimg,'%d'%_gestureId,(30,30), font, 1,(255,255,255),1)
            cv2.imshow('demo',drawimg)
            cv2.waitKey(1)

def CreateInferenceModelWithNN():
    with tf.device('/gpu:%d' % 0):
        with tf.name_scope('%s_%d' % ('dextro-research', 0)) as scope:
            with tf.variable_scope('var_name') as var_scope:
                weights = {
                  'wc1': _variable_on_cpu('wc1', [3, 3, 3, 3, 64], 0.0005),
                  'wc2': _variable_on_cpu('wc2', [3, 3, 3, 64, 128], 0.0005),
                  'wc3a': _variable_on_cpu('wc3a', [3, 3, 3, 128, 256], 0.0005),
                  'wc4a': _variable_on_cpu('wc4a', [3, 3, 3, 256, 512], 0.0005),
                  'wc5a': _variable_on_cpu('wc5a', [3, 3, 3, 512, 512], 0.0005),
                  'wd1': _variable_on_cpu('wd1', [12288, 2048], 0.0005),
                  'wd2': _variable_on_cpu('wd2', [2048, 2048], 0.0005),
                  'out': _variable_on_cpu('wout', [2048, C3DModel.NUM_CLASSES], 0.0005)
                  }
                biases = {
                  'bc1': _variable_on_cpu('bc1', [64], 0.000),
                  'bc2': _variable_on_cpu('bc2', [128], 0.000),
                  'bc3a': _variable_on_cpu('bc3a', [256], 0.000),
                  'bc4a': _variable_on_cpu('bc4a', [512], 0.000),
                  'bc5a': _variable_on_cpu('bc5a', [512], 0.000),
                  'bd1': _variable_on_cpu('bd1', [2048], 0.000),
                  'bd2': _variable_on_cpu('bd2', [2048], 0.000),
                  'out': _variable_on_cpu('bout', [C3DModel.NUM_CLASSES], 0.000),
                  }
            _X = tf.placeholder(tf.float32, shape=(1,
                                                   C3DModel.NUM_FRAMES_PER_CLIP,
                                                   C3DModel.HEIGHT,
                                                   C3DModel.WIDTH,
                                                   C3DModel.CHANNELS))
            logits = C3DModel.inference_c3d(_X, 1, 1, weights, biases)
            probs = tf.nn.softmax(logits)
    return _X,probs



def GetGestureFromVideo(videoBuffer=[],frameNum=16):
    global _gestureId
    _X,predict = CNNLSTM.inference_resnet_lstm(batchsize=1,
            time_steps=6,
            hidden_size=150,
            classes=19,
            loss='softmax_loss',
            train_phase=False)
    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    sess.run(init)
    saver.restore(sess,'./models_lstm/models0/resnet50_lstm_model_best')

    while True:
        if cv2.waitKey(1) and 0xFF==ord('q'):
            break
        if len(videoBuffer)<frameNum:
            continue
        start_time = time.time()
        videoSeqs = []
        if len(videoBuffer)>=frameNum:
            for i in range(6):
                index = int(i*1.0*len(videoBuffer)/6)
                if index>=len(videoBuffer):
                    index = len(videoBuffer)-1
                videoSeqs.append(videoBuffer[index])
        videoArr = np.array(videoSeqs,dtype=np.float32)
        videoArr = videoArr.reshape([-1,CNNLSTM.HEIGHT,CNNLSTM.WIDTH,CNNLSTM.CHANNELS])
        videoArr-=128
        videoArr/=128.0
        predicts = sess.run(predict,feed_dict={
                    _X: videoArr})

        print(predicts)
        _gestureId = predicts

        print('Judge %.2f  per time.'%(time.time()-start_time))
    sess.close()

def Main():
    global _videoBuffer
    videoIOThread = threading.Thread(target=VideoRealtimeProcess,args=(_videoBuffer,32))
    videoIOThread.start()
    showThread = threading.Thread(target=WindowsRealtimeShow)
    showThread.start()
    processThread = threading.Thread(target=GetGestureFromVideo,args=(_videoBuffer,32))
    processThread.start()



if __name__ == '__main__':
    Main()








