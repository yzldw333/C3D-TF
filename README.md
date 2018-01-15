# C3D-TF
C3D-tensorflow version, used for my gesture training.

# model
There is a c3d-model transformed from c3d-facebook project model file.(5 conv 2 fc, little like vgg).
Here is the file.
[c3d.model](https://pan.baidu.com/s/1i5zhPoL) Download and store it in the root folder.
[resnet50_weights_tf_dim_ordering_tf_kernels.h5](https://pan.baidu.com/s/1mjG0hZE) Download and store it in the root folder.
# file directory
- ./ -> root
    - DataSet -> dataProcessScripts
        - opticalflow_gpu/ gpu version
        - GetTrain.py -> data balance. input:label.txt output:gen_label_train.txt gen_label_test.txt
        - OpticalFlow.py -> opticalflow opencv
        - data_augment.py -> offline data augmentation
        - extract_video_frame_2_dirs.py -> origin dataset process
        - generate_label.py
        - shuffle_label.py -> label shuffle
    - Net -> model definition(C3D,ResNet,MobileNet,LSTM)
        - C3DModel.py -> C3D model
        - CNNLSTM.py -> CNN+LSTM interface
        - LoadPCKModel.py -> load params for C3D
        - mobilenet.py -> my mobilenet define
        - mobilenet_v1.py -> official slim definition
        - resnet50.py -> my resnet50 define
        - utils.py -> some common interface for tf
    - demo.py -> demo for real-time video process
    - input_data.py -> data process when training or testing
    - preprocess.py -> img distortion function, which is not used now.
    - test.py -> VIVA Gesture dataset test
    - train_c3d_ucf101.py -> c3d training process
    - train_lstm.py/train_resnet_lstm.py -> train resnet+lstm training process
    - train_mobilenet_lstm.py -> mobilenet+lstm training process

# How to use my scripts?
1. if you use VIVA Gesture Dataset, use scripts under DataSet to perform offline dataset process, otherwise, you should write codes yourself to process Dataset, and use function in extract_video_frame_2_dirs.py
2. Then use train_*.py to train our models. If you want to change the model, please refer CNNLSTM.py and resnet50.py
