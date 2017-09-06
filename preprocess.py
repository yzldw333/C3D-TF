import cv2
import numpy as np

def image_distortion(img_seqs,alpha=6,sig=10):
    '''
    sig is usually selected from 4 to 34.
    '''
    height,width,channel = img_seqs[0].shape
    # compute newY
    deviation_arr = np.random.rand(height,width,1)*2-1
    gaus_deviation_arr = cv2.GaussianBlur(deviation_arr,(3,3),sig)
    gaus_deviation_arr = gaus_deviation_arr.reshape([height,width])*alpha
    src_pos_arr_y = []
    for i in range(height):
        src_pos_arr_y.append(np.array([i]*width))
    src_pos_arr_y = np.array(src_pos_arr_y).astype(np.float32)
    new_pos_arr_y = gaus_deviation_arr+src_pos_arr_y

    # compute newX
    deviation_arr = np.random.rand(height,width,1)*2-1
    gaus_deviation_arr = cv2.GaussianBlur(deviation_arr,(3,3),sig)
    gaus_deviation_arr = gaus_deviation_arr.reshape([height,width])*alpha
    src_pos_arr_x = []
    for i in range(width):
        src_pos_arr_x.append(np.array([i]*height))
    src_pos_arr_x = np.array(src_pos_arr_x).astype(np.float32).transpose()
    new_pos_arr_x = gaus_deviation_arr+src_pos_arr_x

    new_img_seqs = []
    for img in img_seqs:
        new_img = np.zeros(shape=[height,width,channel],dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                newX = new_pos_arr_x[i,j]
                newY = new_pos_arr_y[i,j]
                newValue = Compute_Bilinear_intersection(img,newX,newY).reshape([1,1,channel])
                new_img[i,j,:]=newValue
        new_img_seqs.append(new_img)
        cv2.imshow('t',new_img)
        cv2.waitKey(0)
    return new_img_seqs

def Compute_Bilinear_intersection(srcImg,posX,posY):
    height,width,channel = srcImg.shape
    left = int(posX)
    right = int(posX + 1)
    up = int(posY)
    down = int(posY + 1)
    if right>width-1:
        right=width-1
    if down>height-1:
        down=height-1
    if posX<0:
        left = 0
        right = 0
    if posX>=width:
        left = width-1
        right = width-1
    if posY<0:
        up = 0
        down = 0
    if posY>=height:
        up = height-1
        down = height-1

    if right-left==0:
        hrate = 0
    else:
        hrate = (posX-left)*1.0/(right-left)
    if down-up == 0:
        vrate = 0
    else:
        vrate = (posY-up)*1.0/(down-up)
    value1 = srcImg[up,left,:]+(srcImg[up,right,:]-srcImg[up,left,:])*hrate
    value2 = srcImg[down,left,:]+(srcImg[down,right,:]-srcImg[down,left,:])*hrate
    value = value1+(value2-value1)*vrate
    return value

if __name__ == '__main__':
    img = cv2.imread(r'D:\test.jpg')
    # 获取图像尺寸
    (h, w) = img.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, 30, 1)
    shift = cv2.warpAffine(img,np.float32([[1,0,30],[0,1,0]]),(w,h))
    cv2.imshow('t',shift)
    cv2.waitKey(0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('t',rotated)
    cv2.waitKey(0)
    imglst = [img]
    a = image_distortion(imglst)