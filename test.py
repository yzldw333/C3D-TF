import sys,time,os
sys.path.append('Net')
import CNNLSTM
import mobilenet
from mobilenet import *
from CNNLSTM import *
import input_data
import tensorflow as tf
import numpy as np,time,os
model_filename="./models_lstm/models0/resnet50_lstm_model_0.93-10399"
# predefine
batchsize=2
time_steps=6
hidden_size=150
classes=19
predict_arr = None
def test(valid_root,valid_txt):
    '''
        test function
        params:
        valid_root: dataset root
        valid_txt:  data label file
    '''
    global model_filename
    global predict_arr
    if model_filename=="":
        print("model not exists")
        return None

    graph = tf.Graph()
    with graph.as_default():
        X,Y,predict,loss,accuracy = inference_resnet_lstm(batchsize=batchsize,
        #X,Y,endpoints,features,predict,loss,accuracy = inference_resnet_lstm(batchsize=batchsize,
                time_steps=time_steps,
                hidden_size=hidden_size,
                classes=classes,
                loss='softmax_loss',
                train_phase=True)
        #tf.summary.image('input_images', X, 4)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=15,keep_checkpoint_every_n_hours=1)

        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
            ),graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_filename)
            # train process
            lines=None
            epoch_steps=0
            # test
            test_batch_start = -1
            sum_acc=0
            total_num=0
            while True:
                test_lines = None
                val_images, val_labels, test_batch_start, _, _,test_lines = input_data.read_clip_and_label(
                    rootdir= valid_root,
                    filename= valid_txt,
                    batch_size=1,
                    lines=test_lines,
                    start_pos=test_batch_start,
                    num_frames_per_clip=time_steps,
                    crop_size=(CNNLSTM.HEIGHT, CNNLSTM.WIDTH),
                    shuffle=False,
                    phase='TEST'
                )
                val_images=np.array([val_images[0,:]]*batchsize,dtype=np.float32)
                val_labels=np.array([val_labels]*batchsize,dtype=np.int64).ravel()
                val_images = val_images.reshape([-1,CNNLSTM.HEIGHT,CNNLSTM.WIDTH,CNNLSTM.CHANNELS])
                [acc,pred] = sess.run(
                    [ accuracy,predict],
                    feed_dict={
                        X: val_images,
                        Y: val_labels
                    })
                print(val_labels[0],pred[0])
                predict_arr[int(val_labels[0]),pred[0]]+=1
                sum_acc+=acc
                total_num+=1
                if test_batch_start == -1:
                    acc = sum_acc*1.0/total_num
                    print('test accuracy: %f'%acc)
                    sess.close()
                    break
        print("Done")

if __name__ == '__main__':
    global model_filename
    global predict_arr
    predict_arr = np.zeros([classes,classes],dtype=int)
    for i in range(8):
        model_filename="./models_lstm/models%d/resnet50_lstm_model_best"%i
        print(model_filename)
        test(valid_root='../VIVA_avi_group/VIVA_avi_part%d/val'%i,valid_txt='../VIVA_avi_group/VIVA_avi_part%d/val.txt'%i)

    for i in range(classes):
        total_num = np.sum(predict_arr[i,:])
        strr=""
        for j in range(classes):
            if j==0:
                strr="%.2f"%(predict_arr[i,j]*1.0/total_num)
            else:
                strr+=" %.2f"%(predict_arr[i,j]*1.0/total_num)
        print(strr)
