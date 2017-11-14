import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
GRAD=1
NORMAL=2
GRAD_NORMAL=3

def Drop_Repeat_Frames(seqs, frameNum):
    '''
    Drop and Repeat Frames to make each video has same frame num.
    :param seqs: video sequences
                numpy array
    :param frameNum: frame num
                int
    :return: new video sequences
                numpy array
    '''
    length = np.size(seqs,0)
    shape = seqs.shape
    if len(shape) == 4 :
        seqs = seqs.reshape([length,shape[-3],shape[-2],shape[-1]])
        counts = shape[-3]*shape[-2]*shape[-1]
    elif len(shape) == 3:
        seqs = seqs.reshape([length,shape[-2],shape[-1],1])
        counts = shape[-2]*shape[-1]
    else:
        print("Shape ERROR!")
        return

    shape = seqs.shape
    newSeqs = []
    for i in range(frameNum):
        idx = (int)(i*1.0/frameNum*length+0.5)
        if idx>=length:
            idx=length-1
        newSeqs.extend(list(seqs[idx].ravel()))
    newSeqs = np.array(newSeqs,dtype=np.float32).ravel()
    length = len(newSeqs)//counts
    if len(shape) == 4:
        seqs = newSeqs.reshape([length,shape[-3],shape[-2],shape[-1]])
    return seqs

def GetVideoSeq(name,color,style,height=100,width=100):
    '''
        use opencv to read video sequences
    :param name: video name like 'xxx.avi'
    :param color: is gray?
    :param style: normal/gradient image
    :return: numpy array with the shape of
            (length,channel,height,width)
            or
            (2,length,channel,height,width)
    '''
    cap = cv2.VideoCapture(name)
    seqShape = None
    if cap.isOpened() == False:
        return None
    seqs = []
    length = 0
    while(True):

        res, frame = cap.read()
        if res:
            if color=='gray':
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if style & NORMAL==NORMAL:
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                if len(frame.shape) == 3:
                    frame = frame.transpose([2,0,1])
                seqs.extend(list(frame.ravel()))
            if style & GRAD == GRAD:
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                grad1 = cv2.Sobel(frame,cv2.CV_64F,1,0)
                grad1 = cv2.convertScaleAbs(grad1)
                if len(grad1.shape)==3:
                    grad1 = grad1.transpose([2,0,1])
                grad2 = cv2.Sobel(frame,cv2.CV_64F,0,1)
                grad2 = cv2.convertScaleAbs(grad2)
                if len(grad2.shape)==3:
                    grad2 = frame.transpose([2,0,1])
                grad = cv2.addWeighted(grad1,0.5,grad2,0.5,0)
                #grad = np.append(grad1.ravel(),grad2.ravel())
                seqs.extend(list(grad.ravel()))
                if len(frame.shape)==3:
                    frame = frame.transpose([2,0,1])

            seqShape = frame.shape
            length += 1
        else:
            break
    seqs = np.array(seqs,dtype = np.float32)
    #only gray image
    if len(seqShape) == 2:
        if style&GRAD_NORMAL==GRAD_NORMAL:
            seqs = seqs.reshape([length,2,seqShape[0],seqShape[1]])
        elif style&GRAD==GRAD:
            seqs = seqs.reshape([length,1,seqShape[0],seqShape[1]])
        elif style&NORMAL==NORMAL:
            seqs = seqs.reshape([length,1, seqShape[0], seqShape[1]])
    if len(seqShape) ==3:
        seqs = seqs.reshape([length,3,height,width])
    if len(seqs.shape)==4:
        seqs = seqs.transpose([0,2,3,1])
    return seqs

def extract_frames_2_dirs(rootdir,outdir):
    lst = os.listdir(rootdir)
    for e in lst:

        if os.path.isdir(os.path.join(rootdir,e))==False and e.endswith('avi'):
            tmpdir =os.path.join(outdir,e[:-4])
            if os.path.exists(tmpdir)==False:
                os.mkdir(tmpdir)
            avipath = os.path.join(rootdir,e)
            seqs = GetVideoSeq(avipath, style=NORMAL, color="nogray", height=128, width=171)

            seqs = Drop_Repeat_Frames(seqs, 16)
            print(seqs.shape)
            index = 0
            for e in seqs:
                outpath = os.path.join(tmpdir,'%06d.jpg'%index)
                plt.imsave(outpath,e)
                index+=1
            print(seqs.shape)

def get_gradient_seqs(avipath,height,width):
    grad_seqs = GetVideoSeq(avipath, style=GRAD_NORMAL, color="gray", height=height, width=width)
    grad_seqs = grad_seqs.astype(np.uint8)
    grad_seqs = np.transpose(grad_seqs, [0, 3, 1, 2])
    return grad_seqs

def get_depth_seqs(avipath,height,width):
    depth_seqs = GetVideoSeq(avipath, style=NORMAL, color="gray", height=height, width=width)
    depth_seqs = np.transpose(depth_seqs, [0, 3, 1, 2])
    return depth_seqs

def get_optical_flow_seqs(avipath,height,width):
    pass

def merge_depth_grad_2_avi(gradpath,depthpath,outpath):
    grad_seqs = get_gradient_seqs(gradpath, height=128, width=171)
    depth_seqs = get_depth_seqs(depthpath, height=128, width=171)
    height, width = grad_seqs.shape[-2:]
    length = grad_seqs.shape[0]
    final_seqs = np.zeros([length, 3, height, width], dtype=np.uint8)
    writer = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),25.0,(width,height))
    seqList = []
    for i in range(length):
        final_seqs[i, :2, :, :] = grad_seqs[i, :, :, :]
        final_seqs[i, 2, :, :] = depth_seqs[i, :, :, :]
        frame = final_seqs[i, :, :, :]
        frame = np.transpose(frame, [1, 2, 0])
        seqList.append(frame)
        writer.write(frame)

def merge_depth_grad_2_video(depthdir,graddir,outdir):
    lst = os.listdir(graddir)
    if os.path.exists(outdir)==False:
        os.mkdir(outdir)
        print('Create '+outdir)
    for e in lst:
        outpath = os.path.join(outdir,e)
        if os.path.isdir(os.path.join(graddir, e)) == False and e.endswith('avi'):
            avipath = os.path.join(graddir, e)
            grad_seqs = get_gradient_seqs(avipath,height=128,width=171)
        else:
            print("%s wrong" % e)
            continue
        if os.path.isdir(os.path.join(depthdir,e)) == False and e.endswith('avi'):
            avipath = os.path.join(depthdir,e)
            depth_seqs = get_depth_seqs(avipath,height=128,width=171)
        else:
            print("%s wrong" % e)
            continue
        if grad_seqs.shape[0]!=depth_seqs.shape[0]:
            print("%s wrong"%e)
            continue
        length = grad_seqs.shape[0]
        height, width = grad_seqs.shape[-2:]
        final_seqs = np.zeros([length,3,height,width],dtype=np.uint8)
        #writer = cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc('M','J','P','G'),25,(width,height))
        seqList = []
        for i in range(length):
            final_seqs[i,:2,:,:]=grad_seqs[i,:,:,:]
            final_seqs[i,2,:,:]=depth_seqs[i,:,:,:]
            frame = final_seqs[i,:,:,:]
            frame = np.transpose(frame,[1,2,0])
            seqList.append(frame)
            #cv2.imshow('t',frame)
            #cv2.waitKey()
            #writer.write(frame)
        sp = seqList[0].shape
        seqList = np.array(seqList,dtype=np.uint8)
        print(seqList.shape)
        print('changed to..')
        seqList = Drop_Repeat_Frames(seqList,32)
        print(seqList.shape)
        #writer.release()
        #depth_seqs = Drop_Repeat_Frames(grad_seqs, 16)
        #print(grad_seqs.shape)
        index = 0
        tmpdir = outpath[:-4]
        if os.path.exists(tmpdir) == False:
            os.mkdir(tmpdir)
        for e in seqList:
            jpgpath = os.path.join(tmpdir, '%06d.jpg' % index)
            cv2.imwrite(jpgpath,e)
            index += 1

def CalWriteOpticalFLowAndRGBDifference(rootdir,outdir):
    from DataSet.OpticalFlow import calOpticalFlowAndRGBDifference
    lst = os.listdir(rootdir)
    for e in lst:
        path = os.path.join(rootdir,e)
        if os.path.isfile(path) and e.endswith('avi'):
            cam = cv2.VideoCapture(path)
            ret, frame1 = cam.read()
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            prvs = cv2.resize(prvs,(171,128))
            buffers = [prvs]
            while True:
                ret, frame2 = cam.read()
                if ret==False:
                    break
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                next = cv2.resize(next, (171, 128))
                buffers.append(next)
            length = len(buffers)
            outpath = os.path.join(outdir,e[:-4])
            if os.path.exists(outpath)==False:
                os.mkdir(outpath)
            for i in range(length):
                if i==0:
                    prvs=buffers[i]
                else:
                    next=buffers[i]
                    combine=calOpticalFlowAndRGBDifference(prvs,next)
                    prvs = next
                    cv2.imwrite(os.path.join(outpath,'%06d.jpg'%(i-1)),combine)
        print(e)
if __name__ == '__main__':
    CalWriteOpticalFLowAndRGBDifference(r'E:\dataset\VIVA',r'E:\dataset\VIVA_time')
    #merge_depth_grad_2_video(r'E:\dataset\VIVA\depth',r'E:\dataset\VIVA',r'E:\dataset\VIVA_avi')
    #merge_depth_grad_2_avi(r'E:\dataset\VIVA\01_01_01.avi',r'E:\dataset\VIVA\depth\01_01_01.avi',r'01_01_01_new.avi')
    #extract_frames_2_dirs(r'E:\dataset\VIVA_avi',r'E:\dataset\VIVA_mixjpg')
    #img = cv2.imread(r'E:\dataset\VIVA_mixjpg\01_01_01\000000.jpg')
    #cv2.imshow('t',img[:,:,0])
    #cv2.waitKey()