import tensorflow as tf

# classes
NUM_CLASSES = 19

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
WIDTH = 171
HEIGHT = 128
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

"-----------------------------------------------------------------------------------------------------------------------"

def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME',name=name),
          b
          )

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def inference_c3d(_X, _keep_prob, batch_size, _weights, _biases):

  # Convolution Layer
  with tf.name_scope('C3D_Network'):
    #with tf.name_scope('attention') as scope:
    #  _X = tf.transpose(_X,perm=[0,4,2,3,1])
    #  conv0 = conv3d('conv0', _X, _weights['wc0'], _biases['bc0'])
    #  conv0 = tf.nn.relu(conv0, 'relu1')
    #  conv0 = tf.transpose(conv0,perm=[0,4,2,3,1])

    conv0 = _X
    with tf.name_scope('conv1') as scope:
      conv1 = conv3d('conv1',conv0, _weights['wc1'], _biases['bc1'])
      conv1 = tf.nn.relu(conv1, 'relu1')
      pool1 = max_pool('pool1', conv1, k=1)

  # Convolution Layer
    with tf.name_scope('conv2') as scope:
        conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

  # Convolution Layer
    with tf.name_scope('conv3') as scope:
        conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        pool3 = max_pool('pool3', conv3, k=2)

  # Convolution Layer
    with tf.name_scope('conv4') as scope:
        conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        pool4 = max_pool('pool4', conv4, k=2)

  # Convolution Layer
    with tf.name_scope('conv5') as scope:
        conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        pool5 = max_pool('pool5', conv5, k=2)

  # Fully connected layer
    with tf.name_scope('dense_layer') as scope:
        pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
        dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
        dense1 = tf.nn.dropout(dense1, _keep_prob)
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
        dense2 = tf.nn.dropout(dense2, _keep_prob)

  # Output: class prediction
    with tf.name_scope('logit') as scope:
      out = tf.matmul(dense2, _weights['out']) + _biases['out']
  return out