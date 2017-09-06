import os
import cv2
def inverse(srcroot,dstroot):
    lst = os.listdir(srcroot)
    inverseList={1:2,2:1,3:4,4:3,13:14,14:13,15:16,16:15,27:28,28:27,29:30,30:29,31:32,32:31}
    if os.path.exists(dstroot)==False:
        os.mkdir(dstroot)
    for e in lst:
        path = os.path.join(srcroot,e)
        man = int(e.split('_')[0])
        label = int(e.split('_')[1])
        if label in inverseList:
            newdir='%02d_%02d_%02d'%(man,inverseList[label],4) #4 is inverse index, origin index 1 2 3
            dstpath = os.path.join(dstroot,newdir)
            if os.path.exists(dstpath)==False:
                os.mkdir(dstpath)
            maxjpgnum = 32
            for i in range(maxjpgnum):
                srcimg = os.path.join(path,'%06d.jpg'%i)
                dstimg = os.path.join(dstpath,'%06d.jpg'%(maxjpgnum-1-i))
                cmd = r'copy %s %s'%(srcimg,dstimg)
                os.system(cmd)

def mirror(srcroot,dstroot):
    lst = os.listdir(srcroot)
    if os.path.exists(dstroot)==False:
        os.mkdir(dstroot)
    mirrorList = {1: 2, 2: 1, 3: 3, 4: 4, 13: 14, 14: 13, 15: 15, 16: 16, 27: 27, 28: 28, 29: 30, 30: 29, 31: 31,
                   32: 32}
    for e in lst:
        path = os.path.join(srcroot, e)
        man = int(e.split('_')[0])
        label = int(e.split('_')[1])
        if label in mirrorList:
            if man %2 ==0:
                man = 17 #mirror right hand
            else:
                man = 18 #mirror left hand
            mirrorindex = 5
            newdir = '%02d_%02d_%02d' % (man, mirrorList[label], mirrorindex)  # 5 is mirror index, origin index 1 2 3, inverse 4
            dstpath = os.path.join(dstroot, newdir)
            while os.path.exists(dstpath) == True:
                mirrorindex+=1
                newdir = '%02d_%02d_%02d' % (man, mirrorList[label], mirrorindex)
                dstpath = os.path.join(dstroot, newdir)
            os.mkdir(dstpath)
            maxjpgnum = 32
            for i in range(maxjpgnum):
                srcimg = os.path.join(path, '%06d.jpg' % i)
                dstimg = os.path.join(dstpath, '%06d.jpg' % i)
                img = cv2.imread(srcimg)
                img = cv2.flip(img,1)
                cv2.imwrite(dstimg,img)

def _data_augment(index=0):
    print('data %d is being augmented ...'%index)
    inverse(r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\train'%index,r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\VIVA_jpg_aug'%index)
    mirror(r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\train'%index,r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\VIVA_jpg_mirror'%index)
    mirror(r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\VIVA_jpg_aug'%index,r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\VIVA_jpg_mirror'%index)


if __name__ == '__main__':
    import threading

    #data_augment(0)
    #data_augment(1)
    #data_augment(2)
    #data_augment(3)
    t1 = threading.Thread(target=_data_augment,kwargs={'index':3})
    t1.start()
    t2 = threading.Thread(target=_data_augment,kwargs={'index':4})
    t2.start()
    t3 = threading.Thread(target=_data_augment,kwargs={'index':5})
    t3.start()
    t4 = threading.Thread(target=_data_augment,kwargs={'index':6})
    t4.start()
    t5 = threading.Thread(target=_data_augment,kwargs={'index':7})
    t5.start()


