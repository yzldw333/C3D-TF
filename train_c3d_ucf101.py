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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from Net import C3DModel
from Net import LoadPCKModel
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 70000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
FLAGS = flags.FLAGS

MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  with tf.name_scope('input') as scope:
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           C3DModel.NUM_FRAMES_PER_CLIP,
                                                           C3DModel.HEIGHT,
                                                           C3DModel.WIDTH,
                                                           C3DModel.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))

    visualize_imgs = tf.slice(images_placeholder,[0,0,0,0,0],
                                                  [1, C3DModel.NUM_FRAMES_PER_CLIP,
                                                   C3DModel.HEIGHT,
                                                   C3DModel.WIDTH,
                                                   C3DModel.CHANNELS]
                              )
    visualize_imgs = tf.reshape(visualize_imgs,[C3DModel.NUM_FRAMES_PER_CLIP,
                                                   C3DModel.HEIGHT,
                                                   C3DModel.WIDTH,
                                                   C3DModel.CHANNELS])
    tf.summary.image('input_images',visualize_imgs,16)
  return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels)
                  )
  tf.summary.scalar(
                  name_scope + 'cross entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
  tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)
  tf.add_to_collection('losses', cross_entropy_mean)
  losses = tf.get_collection('losses', name_scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  tf.summary.scalar(name_scope + 'total loss', total_loss)

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss

def cross_entropy_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels)
    )
    tf.summary.scalar(
        name_scope + '-cross_entropy',
        cross_entropy_mean
    )
    return cross_entropy_mean

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)
  use_pretrained_model = True
  model_filename = "./models/c3d_ucf_model-21000"
  #model_filename = ""
  if len(model_filename)!=0:
    start_steps=int(model_filename.strip().split('-')[-1])
  else:
    start_steps=0
  pckmodel_filename = "./c3d.model"
  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    base_lr = 0.001
    learning_rate = tf.Variable(base_lr,trainable=False)
    #opt1 = tf.train.AdamOptimizer(learning_rate)
    #opt2 = tf.train.AdamOptimizer(learning_rate*2)
    opt1 = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
    opt2 = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        with tf.name_scope('%s-%d' % ('ludongwei-pc', gpu_index)) as scope:
          with tf.variable_scope('var_name') as var_scope:
            weights = {
              #'wc0': _variable_on_cpu('wc0', [1, 1, 1, 16,16], tf.constant_initializer(1/16.0)),
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wd1': _variable_with_weight_decay('wd1', [12288, 2048], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [2048, 2048], 0.0005),
              'out': _variable_with_weight_decay('wout', [2048, C3DModel.NUM_CLASSES], 0.0005)
              }
            biases = {
              #'bc0': _variable_with_weight_decay('bc0', [16],0.000),
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [2048], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [2048], 0.000),
              'out': _variable_with_weight_decay('bout', [C3DModel.NUM_CLASSES], 0.000),
              }
          varlist1 = weights.values()
          varlist2 = biases.values()
          logit = C3DModel.inference_c3d(
                          images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                          0.5,
                          FLAGS.batch_size,
                          weights,
                          biases
                          )
          loss = cross_entropy_loss(
                          scope,
                          logit,
                          labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                          )
          tf.summary.scalar(name='loss', tensor=loss)
          grads1 = opt1.compute_gradients(loss, varlist1)
          grads2 = opt2.compute_gradients(loss, varlist2)
          tower_grads1.append(grads1)
          tower_grads2.append(grads2)
          logits.append(logit)
          tf.get_variable_scope().reuse_variables()
    logits = tf.concat(0, logits)

    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    apply_gradient_op1 = opt1.apply_gradients(grads1)
    apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
    #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2)#, variables_averages_op)
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(
                                    allow_soft_placement=True
                                    #log_device_placement=True,
                                    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
                                    ),graph=graph
                    )
    sess.run(init)
    if os.path.isfile(model_filename+'.meta') and use_pretrained_model:
      saver.restore(sess, model_filename)
      print(model_filename)
      pass
    elif os.path.isfile(pckmodel_filename) and use_pretrained_model:
      print(pckmodel_filename)
      LoadPCKModel.load(sess,weights,biases,pckmodel_filename)
      pass

    # Create summary writter

    merged = tf.summary.merge_all()

    sess.run(tf.assign(learning_rate,0.0001))

    train_writer = tf.summary.FileWriter('./visual_logs/train/attention', sess.graph)
    next_batch_start = -1
    last_acc = 0
    lines=None
    epoch = 0
    for step in xrange(start_steps,FLAGS.max_steps):
        start_time = time.time()

        train_images, train_labels, next_batch_start, _, _,lines = input_data.read_clip_and_label(
                        rootdir = 'E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\train',
                        filename='E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\gen_train_shuffle.txt',
                        batch_size=FLAGS.batch_size * gpu_num,
                        lines=lines,
                        start_pos=next_batch_start,
                        num_frames_per_clip=C3DModel.NUM_FRAMES_PER_CLIP,
                        crop_size=(C3DModel.HEIGHT,C3DModel.WIDTH),
                        shuffle=False,
                        phase='TRAIN'
                        )
        _,losses,summary = sess.run([train_op,loss,merged], feed_dict={
                        images_placeholder: train_images,
                        labels_placeholder: train_labels
                        })
        train_writer.add_summary(summary, step)
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        print ("lr:%f "%sess.run(learning_rate)+"loss: " + "{:.4f}".format(losses))
        # Save a checkpoint and evaluate the model periodically.
        if step%1000==0 and step!=0 and step!=start_steps or step+1 == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)
            print('Model Saved.')
            print('Training Data Eval:')
        if next_batch_start == -1:
            epoch+=1
            if epoch%40==0 and epoch!=0:
                # test
                test_batch_start = -1
                sum_acc=0
                total_num=0
                while True:
                    test_lines = None
                    val_images, val_labels, test_batch_start, _, _,test_lines = input_data.read_clip_and_label(
                        rootdir='E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\val',
                        filename='E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\val.txt',
                        batch_size=1 * gpu_num,
                        lines=test_lines,
                        start_pos=test_batch_start,
                        num_frames_per_clip=C3DModel.NUM_FRAMES_PER_CLIP,
                        crop_size=(C3DModel.HEIGHT, C3DModel.WIDTH),
                        shuffle=False,
                        phase='TEST'
                    )
                    val_images=np.array([val_images[0,:]]*FLAGS.batch_size,dtype=np.float32)
                    val_labels=np.array([val_labels]*FLAGS.batch_size,dtype=np.float32).ravel()
                    summary, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={
                            images_placeholder: val_images,
                            labels_placeholder: val_labels
                        })
                    sum_acc+=acc
                    total_num+=1
                    if test_batch_start == -1:
                        acc = sum_acc*1.0/total_num
                        print('Epoch: %d test accuracy: %f'%(epoch,acc))
                        break
                if acc<last_acc*1.1:
                    learning_rate_value = sess.run(learning_rate)
                    learning_rate_value *= 0.5
                    assign_op = tf.assign(learning_rate,learning_rate_value)
                    sess.run(assign_op)
                    print("acc:%f < last_acc*1.1:%f"%(acc,last_acc*1.1))
                    #print("learning_rate changed to %f"%sess.run(learning_rate))
                last_acc = acc
    print("Done")
    train_writer.flush()
    train_writer.close()

def run_testing():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    use_pretrained_model = True
    model_filename = "./models/82p/c3d_ucf_model-66000"
    pckmodel_filename = "./c3d.model"
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        images_placeholder, labels_placeholder = placeholder_inputs(
            1 * gpu_num
        )
        logits = []
        for gpu_index in range(0, gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                with tf.name_scope('%s_%d' % ('dextro-research', gpu_index)) as scope:
                    with tf.variable_scope('var_name') as var_scope:
                        weights = {
                            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                            'wd1': _variable_with_weight_decay('wd1', [12288, 2048], 0.0005),
                            'wd2': _variable_with_weight_decay('wd2', [2048, 2048], 0.0005),
                            'out': _variable_with_weight_decay('wout', [2048, C3DModel.NUM_CLASSES], 0.0005)
                        }
                        biases = {
                            'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                            'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                            'bd1': _variable_with_weight_decay('bd1', [2048], 0.000),
                            'bd2': _variable_with_weight_decay('bd2', [2048], 0.000),
                            'out': _variable_with_weight_decay('bout', [C3DModel.NUM_CLASSES], 0.000),
                        }
                    logit = C3DModel.inference_c3d(
                        images_placeholder[gpu_index * 1:(gpu_index + 1) * 1, :, :, :, :],
                        1,
                        1,
                        weights,
                        biases
                    )
                    logits.append(logit)
                    tf.get_variable_scope().reuse_variables()
        logits = tf.concat(0, logits)
        predict_value = tf.argmax(logit, 1)
        accuracy = tower_acc(logits, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True
                #log_device_placement=True
            ), graph=graph
        )
        sess.run(init)
        if os.path.isfile(model_filename + '.meta') and use_pretrained_model:
            saver.restore(sess, model_filename)
            print("Model Reading!")
            pass
        elif os.path.exists(pckmodel_filename) and use_pretrained_model:
            LoadPCKModel.load(weights, biases, pckmodel_filename)

        # Create summary writter

        merged = tf.summary.merge_all()

        test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)
        next_batch_start = -1
        sum_acc=0
        total_num=0
        lines=None
        predictDict={}
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (step, duration))

            val_images, val_labels, next_batch_start, _, _,lines = input_data.read_clip_and_label(
                rootdir='E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\val',
                filename='E:\\dataset\\VIVA_avi_group\\VIVA_avi_part0\\val.txt',
                batch_size=1 * gpu_num,
                lines=lines,
                start_pos=next_batch_start,
                num_frames_per_clip=C3DModel.NUM_FRAMES_PER_CLIP,
                crop_size=(C3DModel.HEIGHT, C3DModel.WIDTH),
                shuffle=False,
                phase='TEST'
            )
            summary, acc ,predict= sess.run(
                [merged, accuracy,predict_value],
                feed_dict={
                    images_placeholder: val_images,
                    labels_placeholder: val_labels
                })
            sum_acc+=acc
            total_num+=1
            print("accuracy: " + "{:.5f}".format(acc))
            print("predict: %d, label: %d"%(predict,val_labels[0]))
            if val_labels[0] not in predictDict:
                predictDict[val_labels[0]]=[1,0]
            else:
                predictDict[val_labels[0]][0]+=1
            if predict==val_labels[0]:
                predictDict[val_labels[0]][1]+=1
            test_writer.add_summary(summary, step)
            if next_batch_start==-1:
                break
        print("accuracy: " + "{:.5f}".format(sum_acc*1.0/total_num))
        for key in predictDict:
            value = predictDict[key]
            print("class %d, predict Accuracy:%f"%(key,value[1]/value[0]))
        test_writer.flush()
        test_writer.close()
    print("done")
def main(_):
  #run_testing()
  run_training()

if __name__ == '__main__':
  tf.app.run()
