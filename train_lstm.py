import sys,time,os
sys.path.append('Net')
import CNNLSTM
import resnet50
from resnet50 import *
from CNNLSTM import *
import input_data
import tensorflow as tf
import numpy as np,time,os
import cv2
# predefine
model_save_dir="./models"
use_pretrained_model=True
batchsize=6
time_steps=4
hidden_size=128
classes=19
max_steps=70000


def train(train_root,train_txt,valid_root,valid_txt):
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)
  model_filename = "./models/resnet50_lstm_model-41000"
  model_filename = ""
  if len(model_filename)!=0:
    start_steps=int(model_filename.strip().split('-')[-1])
  else:
    start_steps=0
  graph = tf.Graph()
  with graph.as_default():
    X,Y,predict,loss,accuracy = inference_resnet_lstm(batchsize=batchsize,time_steps=time_steps,hidden_size=hidden_size,classes=classes)
    learning_rate_value = 0.0001
    learning_rate = tf.Variable(learning_rate_value, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver(max_to_keep=10,keep_checkpoint_every_n_hours=1)
    init = tf.global_variables_initializer()
    load_op = load_pretrained_model_ops()
    #tf.summary.image('input_images', X, 4)
    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
        ),graph=graph) as sess:
        sess.run(init)
        if model_filename == "":
            sess.run(load_op)
            pass
        else:
            saver.restore(sess, model_filename)

        train_writer = tf.summary.FileWriter('./visual_logs/train/attention', sess.graph)
        # train process

        next_batch_start = -1
        last_acc = 0
        lines=None
        epoch = int(start_steps/(1900/(batchsize)))
        for step in range(start_steps,max_steps):
            start_time = time.time()
            if epoch>=0:
                # open data augmentation
                status = 'TRAIN'
            else:
                # close data augmentation
                status = 'TEST' 
            startprocess_time = time.time()
            train_images, train_labels, next_batch_start, _, _,lines = input_data.read_clip_and_label(
                            rootdir = train_root,
                            filename= train_txt,
                            batch_size=batchsize,
                            lines=lines,
                            start_pos=next_batch_start,
                            num_frames_per_clip=CNNLSTM.NUM_FRAMES_PER_CLIP,
                            crop_size=(CNNLSTM.HEIGHT,CNNLSTM.WIDTH),
                            shuffle=False,
                            phase=status
                            )
            for i in range(CNNLSTM.NUM_FRAMES_PER_CLIP):
                cv2.imshow('t',train_images[0,i,:,:,:])
                cv2.waitKey(0)
            train_images = train_images.reshape([-1,CNNLSTM.HEIGHT,CNNLSTM.WIDTH,CNNLSTM.CHANNELS])
            endprocess_time = time.time()
            preprocess_time = ((endprocess_time-startprocess_time)/(batchsize*1))
            print("preprocess per time :%f"%preprocess_time)
            _,losses = sess.run([train_op,loss], feed_dict={
                            X: train_images,
                            Y: train_labels
                            })
            ##train_writer.add_summary(summary, step)
            duration = time.time() - start_time
            print('Epoch: %d Step %d: %.3f sec' % (epoch, step, duration))
            print ("lr:%f loss: %.4f"%(sess.run(learning_rate),losses))
            # Save a checkpoint and evaluate the model periodically.
            if step%2000==0 and step!=0 and step!=start_steps or step+1 == max_steps:
                saver.save(sess, os.path.join(model_save_dir, 'resnet50_lstm_model'), global_step=step)
                print('Model Saved.')
                print('Training Data Eval:')
            if next_batch_start == -1:
                epoch+=1
                if epoch==200:
                    print("Learning Done.")
                    break

                if epoch%10==0 and epoch!=0:
                    # test
                    test_batch_start = -1
                    sum_acc=0
                    total_num=0
                    patients=0
                    while True:
                        test_lines = None
                        val_images, val_labels, test_batch_start, _, _,test_lines = input_data.read_clip_and_label(
                            rootdir= valid_root,
                            filename= valid_txt,
                            batch_size=1,
                            lines=test_lines,
                            start_pos=test_batch_start,
                            num_frames_per_clip=CNNLSTM.NUM_FRAMES_PER_CLIP,
                            crop_size=(CNNLSTM.HEIGHT, CNNLSTM.WIDTH),
                            shuffle=False,
                            phase='TEST'
                        )
                        val_images=np.array([val_images[0,:]]*batchsize,dtype=np.float32)
                        val_labels=np.array([val_labels]*batchsize,dtype=np.float32).ravel()
                        val_images = val_images.reshape([-1,CNNLSTM.HEIGHT,CNNLSTM.WIDTH,CNNLSTM.CHANNELS])
                        [acc] = sess.run(
                            [ accuracy],
                            feed_dict={
                                X: val_images,
                                Y: val_labels
                            })
                        sum_acc+=acc
                        total_num+=1
                        if test_batch_start == -1:
                            acc = sum_acc*1.0/total_num
                            print('Epoch: %d test accuracy: %f'%(epoch,acc))
                            break
                    if acc<last_acc*1.02:
                        learning_rate_value = sess.run(learning_rate)
                        patients+=1
                        if patients>=3:
                            learning_rate_value = learning_rate_value*0.5
                            assign_op = tf.assign(learning_rate, learning_rate_value)
                            sess.run(assign_op)
                            print("acc:%f < last_acc*1.1:%f"%(acc,last_acc*1.02))
                            print("learning_rate changed to %f"%sess.run(learning_rate))
                            last_acc = acc
                            patients=0
    print("Done")
    train_writer.flush()
    train_writer.close()




if __name__ == '__main__':
    train(train_root=r'E:/dataset/VIVA_avi_group/VIVA_avi_part0/train',train_txt='E:/dataset/VIVA_avi_group/VIVA_avi_part0/gen_train_shuffle.txt',
        valid_root='E:/dataset/VIVA_avi_group/VIVA_avi_part0/val',valid_txt='E:/dataset/VIVA_avi_group/VIVA_avi_part0/val.txt')

