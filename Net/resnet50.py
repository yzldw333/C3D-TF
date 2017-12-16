from utils import *
def resnet50(X,output_dim=1024,train_phase=True,no_top=False):

    conv1=conv(X,3,64,'conv1',stride_size=2,kernel_size=7,padding='SAME')
    #print('conv1',conv1)
    pooling1=pooling_3x3(conv1,'pooling1')
    #print('pool1',pooling1)
    block1=Block(pooling1,conv1_input_dim=64,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='conv2',bn_name_1='bn1',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='conv3',bn_name_2='bn2',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='conv4',bn_name_3='bn4',
        pass_way_conv_name='pass_way_conv1',train_phase=train_phase,pass_way_bn_name='pass_way_bn1',up_dim=True)

    block2=Block(block1,conv1_input_dim=256,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='conv5',bn_name_1='bn5',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='conv6',bn_name_2='bn6',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='conv7',bn_name_3='bn7',
        pass_way_conv_name='pass_way_conv2',train_phase=train_phase,pass_way_bn_name='pass_way_bn2',up_dim=False)


    block3=Block(block2,conv1_input_dim=256,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='conv8',bn_name_1='bn8',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='conv9',bn_name_2='bn9',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='conv10',bn_name_3='bn10',
        pass_way_conv_name='pass_way_conv3',train_phase=train_phase,pass_way_bn_name='pass_way_bn3',up_dim=False)


    block4 = Block(block3, conv1_input_dim=256, conv1_output_dim=128, stride1_size=2, kernel1_size=1, conv_name_1='conv11',bn_name_1='bn11',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='conv12',bn_name_2='bn12',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='conv13',bn_name_3='bn13',
                   pass_way_conv_name='pass_way_conv4', train_phase=train_phase,pass_way_bn_name='pass_way_bn4', up_dim=True)

    block5 = Block(block4, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='conv14',bn_name_1='bn14',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='conv15',bn_name_2='bn15',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='conv16',bn_name_3='bn16',
                   pass_way_conv_name='pass_way_conv5', train_phase=train_phase,pass_way_bn_name='pass_way_bn5', up_dim=False)

    block6 = Block(block5, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='conv17',bn_name_1='bn17',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='conv18',bn_name_2='bn18',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='conv19',bn_name_3='bn19',
                   pass_way_conv_name='pass_way_conv6', train_phase=train_phase,pass_way_bn_name='pass_way_bn6', up_dim=False)


    block7 = Block(block6, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='conv20',bn_name_1='bn20',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='conv21',bn_name_2='bn21',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='conv22',bn_name_3='bn22',
                   pass_way_conv_name='pass_way_conv7', train_phase=train_phase,pass_way_bn_name='pass_way_bn7', up_dim=False)
    #print('block7',block7)
    block8 = Block(block7, conv1_input_dim=512, conv1_output_dim=256, stride1_size=2, kernel1_size=1, conv_name_1='conv23',bn_name_1='bn23',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv24',bn_name_2='bn24',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv25',bn_name_3='bn25',
                   pass_way_conv_name='pass_way_conv8',train_phase=train_phase, pass_way_bn_name='pass_way_bn8', up_dim=True)
    #print('block8',block8)
    block9 = Block(block8, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='conv26',bn_name_1='bn26',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv27',bn_name_2='bn27',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv28',bn_name_3='bn28',
                   pass_way_conv_name='pass_way_conv9',train_phase=train_phase, pass_way_bn_name='pass_way_bn9', up_dim=False)
    #print('block9',block9)
    block10 = Block(block9, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='conv29',bn_name_1='bn29',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv30',bn_name_2='bn30',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv31',bn_name_3='bn31',
                   pass_way_conv_name='pass_way_conv10',train_phase=train_phase, pass_way_bn_name='pass_way_bn10', up_dim=False)

    #print('block10',block10)
    block11 = Block(block10, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='conv32',bn_name_1='bn32',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv33',bn_name_2='bn33',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv34',bn_name_3='bn34',
                   pass_way_conv_name='pass_way_conv11',train_phase=train_phase, pass_way_bn_name='pass_way_bn11', up_dim=False)

    block12 = Block(block11, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='conv35',bn_name_1='bn35',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv36',bn_name_2='bn36',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv37',bn_name_3='bn37',
                   pass_way_conv_name='pass_way_conv12', train_phase=train_phase,pass_way_bn_name='pass_way_bn12', up_dim=False)

    #print('block12',block12)
    block13 = Block(block12, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='conv38',bn_name_1='bn38',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='conv39',bn_name_2='bn39',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='conv40',bn_name_3='bn40',
                   pass_way_conv_name='pass_way_conv13', train_phase=train_phase,pass_way_bn_name='pass_way_bn13', up_dim=False)

    block14 = Block(block13, conv1_input_dim=1024, conv1_output_dim=512, stride1_size=2, kernel1_size=1, conv_name_1='conv41',bn_name_1='bn41',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='conv42',bn_name_2='bn42',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='conv43',bn_name_3='bn43',
                   pass_way_conv_name='pass_way_conv14',train_phase=train_phase, pass_way_bn_name='pass_way_bn14', up_dim=True)

    block15 = Block(block14, conv1_input_dim=2048, conv1_output_dim=512, stride1_size=1, kernel1_size=1, conv_name_1='conv44',bn_name_1='bn44',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='conv45',bn_name_2='bn45',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='conv46',bn_name_3='bn46',
                   pass_way_conv_name='pass_way_conv15',train_phase=train_phase, pass_way_bn_name='pass_way_bn15', up_dim=False)

    block16 = Block(block15, conv1_input_dim=2048, conv1_output_dim=512, stride1_size=1, kernel1_size=1, conv_name_1='conv47',bn_name_1='bn47',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='conv48',bn_name_2='bn48',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='conv49',bn_name_3='bn49',
                   pass_way_conv_name='pass_way_conv16',train_phase=train_phase, pass_way_bn_name='pass_way_bn16', up_dim=False)

    ave_pool=tf.nn.avg_pool(block16,ksize=[1,7,7,1],strides=[1,1,1,1],padding='VALID')

    ave_shape=ave_pool.get_shape().as_list()
    ave_pool=tf.reshape(ave_pool,shape=[-1,ave_shape[3]])
    if no_top==True:
        return ave_pool
    else:
        fc_out=fc(ave_pool,input_dim=2048,output_dim=output_dim,name='fc')
        return fc_out,block4

def resnet50_BVLC(X,output_dim=1024,train_phase=True,no_top=False):

    conv1=conv(X,3,64,'conv1',stride_size=2,kernel_size=7,padding='SAME')
    conv1=tf.nn.relu(BatchNorm(conv1,train_phase,'bn_conv1'))

    #print('conv1',conv1)
    pooling1=pooling_3x3(conv1,'pooling1')
    #print('pool1',pooling1)
    block1=Block(pooling1,conv1_input_dim=64,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='res2a_branch2a',bn_name_1='bn2a_branch2a',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='res2a_branch2b',bn_name_2='bn2a_branch2b',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='res2a_branch2c',bn_name_3='bn2a_branch2c',
        pass_way_conv_name='res2a_branch1',train_phase=train_phase,pass_way_bn_name='bn2a_branch1',up_dim=True)

    block2=Block(block1,conv1_input_dim=256,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='res2b_branch2a',bn_name_1='bn2b_branch2a',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='res2b_branch2b',bn_name_2='bn2b_branch2b',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='res2b_branch2c',bn_name_3='bn2b_branch2c',
        pass_way_conv_name='pass_way_conv2',train_phase=train_phase,pass_way_bn_name='pass_way_bn2',up_dim=False)


    block3=Block(block2,conv1_input_dim=256,conv1_output_dim=64,stride1_size=1,kernel1_size=1,conv_name_1='res2c_branch2a',bn_name_1='bn2c_branch2a',
          conv2_input_dim=64,conv2_output_dim=64,stride2_size=1,kernel2_size=3,conv_name_2='res2c_branch2b',bn_name_2='bn2c_branch2b',
          conv3_input_dim=64,conv3_output_dim=256,stride3_size=1,kernel3_size=1,conv_name_3='res2c_branch2c',bn_name_3='bn2c_branch2c',
        pass_way_conv_name='pass_way_conv3',train_phase=train_phase,pass_way_bn_name='pass_way_bn3',up_dim=False)


    block4 = Block(block3, conv1_input_dim=256, conv1_output_dim=128, stride1_size=2, kernel1_size=1, conv_name_1='res3a_branch2a',bn_name_1='bn3a_branch2a',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='res3a_branch2b',bn_name_2='bn3a_branch2b',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='res3a_branch2c',bn_name_3='bn3a_branch2c',
                   pass_way_conv_name='res3a_branch1', train_phase=train_phase,pass_way_bn_name='bn3a_branch1', up_dim=True)

    block5 = Block(block4, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='res3b_branch2a',bn_name_1='bn3b_branch2a',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='res3b_branch2b',bn_name_2='bn3b_branch2b',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='res3b_branch2c',bn_name_3='bn3b_branch2c',
                   pass_way_conv_name='pass_way_conv5', train_phase=train_phase,pass_way_bn_name='pass_way_bn5', up_dim=False)

    block6 = Block(block5, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='res3c_branch2a',bn_name_1='bn3c_branch2a',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='res3c_branch2b',bn_name_2='bn3c_branch2b',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='res3c_branch2c',bn_name_3='bn3c_branch2c',
                   pass_way_conv_name='pass_way_conv6', train_phase=train_phase,pass_way_bn_name='pass_way_bn6', up_dim=False)


    block7 = Block(block6, conv1_input_dim=512, conv1_output_dim=128, stride1_size=1, kernel1_size=1, conv_name_1='res3d_branch2a',bn_name_1='bn3d_branch2a',
                   conv2_input_dim=128, conv2_output_dim=128, stride2_size=1, kernel2_size=3, conv_name_2='res3d_branch2b',bn_name_2='bn3d_branch2b',
                   conv3_input_dim=128, conv3_output_dim=512, stride3_size=1, kernel3_size=1, conv_name_3='res3d_branch2c',bn_name_3='bn3d_branch2c',
                   pass_way_conv_name='pass_way_conv7', train_phase=train_phase,pass_way_bn_name='pass_way_bn7', up_dim=False)
    #print('block7',block7)
    block8 = Block(block7, conv1_input_dim=512, conv1_output_dim=256, stride1_size=2, kernel1_size=1, conv_name_1='res4a_branch2a',bn_name_1='bn4a_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4a_branch2b',bn_name_2='bn4a_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4a_branch2c',bn_name_3='bn4a_branch2c',
                   pass_way_conv_name='res4a_branch1',train_phase=train_phase, pass_way_bn_name='bn4a_branch1', up_dim=True)
    #print('block8',block8)
    block9 = Block(block8, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='res4b_branch2a',bn_name_1='bn4b_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4b_branch2b',bn_name_2='bn4b_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4b_branch2c',bn_name_3='bn4b_branch2c',
                   pass_way_conv_name='pass_way_conv9',train_phase=train_phase, pass_way_bn_name='pass_way_bn9', up_dim=False)
    #print('block9',block9)
    block10 = Block(block9, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='res4c_branch2a',bn_name_1='bn4c_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4c_branch2b',bn_name_2='bn4c_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4c_branch2c',bn_name_3='bn4c_branch2c',
                   pass_way_conv_name='pass_way_conv10',train_phase=train_phase, pass_way_bn_name='pass_way_bn10', up_dim=False)

    #print('block10',block10)
    block11 = Block(block10, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='res4d_branch2a',bn_name_1='bn4d_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4d_branch2b',bn_name_2='bn4d_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4d_branch2c',bn_name_3='bn4d_branch2c',
                   pass_way_conv_name='pass_way_conv11',train_phase=train_phase, pass_way_bn_name='pass_way_bn11', up_dim=False)

    block12 = Block(block11, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='res4e_branch2a',bn_name_1='bn4e_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4e_branch2b',bn_name_2='bn4e_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4e_branch2c',bn_name_3='bn4e_branch2c',
                   pass_way_conv_name='pass_way_conv12', train_phase=train_phase,pass_way_bn_name='pass_way_bn12', up_dim=False)

    #print('block12',block12)
    block13 = Block(block12, conv1_input_dim=1024, conv1_output_dim=256, stride1_size=1, kernel1_size=1, conv_name_1='res4f_branch2a',bn_name_1='bn4f_branch2a',
                   conv2_input_dim=256, conv2_output_dim=256, stride2_size=1, kernel2_size=3, conv_name_2='res4f_branch2b',bn_name_2='bn4f_branch2b',
                   conv3_input_dim=256, conv3_output_dim=1024, stride3_size=1, kernel3_size=1, conv_name_3='res4f_branch2c',bn_name_3='bn4f_branch2c',
                   pass_way_conv_name='pass_way_conv13', train_phase=train_phase,pass_way_bn_name='pass_way_bn13', up_dim=False)

    block14 = Block(block13, conv1_input_dim=1024, conv1_output_dim=512, stride1_size=2, kernel1_size=1, conv_name_1='res5a_branch2a',bn_name_1='bn5a_branch2a',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='res5a_branch2b',bn_name_2='bn5a_branch2b',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='res5a_branch2c',bn_name_3='bn5a_branch2c',
                   pass_way_conv_name='res5a_branch1',train_phase=train_phase, pass_way_bn_name='bn5a_branch1', up_dim=True)

    block15 = Block(block14, conv1_input_dim=2048, conv1_output_dim=512, stride1_size=1, kernel1_size=1, conv_name_1='res5b_branch2a',bn_name_1='bn5b_branch2a',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='res5b_branch2b',bn_name_2='bn5b_branch2b',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='res5b_branch2c',bn_name_3='bn5b_branch2c',
                   pass_way_conv_name='pass_way_conv15',train_phase=train_phase, pass_way_bn_name='pass_way_bn15', up_dim=False)

    block16 = Block(block15, conv1_input_dim=2048, conv1_output_dim=512, stride1_size=1, kernel1_size=1, conv_name_1='res5c_branch2a',bn_name_1='bn5c_branch2a',
                   conv2_input_dim=512, conv2_output_dim=512, stride2_size=1, kernel2_size=3, conv_name_2='res5c_branch2b',bn_name_2='bn5c_branch2b',
                   conv3_input_dim=512, conv3_output_dim=2048, stride3_size=1, kernel3_size=1, conv_name_3='res5c_branch2c',bn_name_3='bn5c_branch2c',
                   pass_way_conv_name='pass_way_conv16',train_phase=train_phase, pass_way_bn_name='pass_way_bn16', up_dim=False)

    ave_pool=tf.nn.avg_pool(block16,ksize=[1,7,7,1],strides=[1,1,1,1],padding='VALID')

    ave_shape=ave_pool.get_shape().as_list()
    ave_pool=tf.reshape(ave_pool,shape=[-1,ave_shape[3]])
    if no_top==True:
        return ave_pool
    else:
        fc_out=fc(ave_pool,input_dim=2048,output_dim=output_dim,name='fc')
        return fc_out,block4

def load_pretrained_resnet50_model_ops(model='resnet50_weights_tf_dim_ordering_tf_kernels.h5'):
    import h5py
    file = h5py.File(model)
    vars = tf.global_variables()
    dct={}
    for e in vars:
        print(e)
        dct[e.name]=e
    ops = []
    for key,value in file.items():
        if 'bn' in key:
            for subkey,value in value.items():
                subkey = subkey[:-2]
                joinkey=''
                if subkey.endswith('gamma'):
                    joinkey = key+'/gamma:0'
                elif subkey.endswith('beta'):
                    joinkey = key+'/beta:0'
                elif subkey.endswith('running_mean'):
                    joinkey = key+'/moving_mean:0'
                elif subkey.endswith('running_std'):
                    joinkey = key+'/moving_variance:0'
                if joinkey in dct:
                    ops.append(tf.assign(dct[joinkey], value.value))

                #print(key,value)
        elif 'conv'or 'fc' in key:
            for key,value in value.items():
                key = key[:-4]+'/'+key[-3:]
                if key not in dct:
                    continue
                ops.append(tf.assign(dct[key],value.value))
                #print(key,value)
    return ops

def test():
    X = tf.placeholder(dtype=tf.float32,shape=[10,224,224,3])
    resnet50_BVLC(X,output_dim=1024,train_phase=True,no_top=False)
    ops = load_pretrained_model_ops(model='resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    print(len(ops))
