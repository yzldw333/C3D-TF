# encoding:utf-8
import numpy as np
import cv2
import cu_tvl1_opticalflow
# import video #Opencv Python自带的读取

help_message = '''''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

'''
pFlow = None
flowBuffer = None

def calOpticalFlowAndRGBDifference(left,right):
    def calTV1OpticalFlow(prvs,next):
        import time
        #st = time.time()
        global pFlow
        global flowBuffer
        if prvs is None or next is None or prvs.shape!=next.shape:
            return None
        #if pFlow is None:
        #    pFlow = cv2.DualTVL1OpticalFlow_create()
        #if flowBuffer is None or flowBuffer.shape != next.shape:
        #    flowBuffer = np.zeros((next.shape[0],next.shape[1],2),dtype=np.float32)
        scale_prvs = cv2.resize(prvs,(prvs.shape[1],prvs.shape[0]))
        scale_next = cv2.resize(next,(next.shape[1],next.shape[0]))
        flowBuffer = cu_tvl1_opticalflow.cudaTVL1OpticalFlowWrapper(scale_prvs,scale_next)
        #flowBuffer = cv2.resize(flowBuffer,(next.shape[1],next.shape[0]))
        #print(time.time()-st)
        #flowBuffer = cv2.convertScaleAbs(flowBuffer)
        #flowBuffer[np.abs(flowBuffer) < 10]=0
        #flowBuffer[:,:,0]-=np.mean(flowBuffer[:,:,0])
        #flowBuffer[:,:,1]-=np.mean(flowBuffer[:,:,1])
        return flowBuffer.copy()
    def calRGBDifference(prvs,next):
        arr = next-prvs
        #arr = cv2.convertScaleAbs(arr)
        return arr
    def norm(arr):
        maxx = np.max(arr)
        minn = np.min(arr)
        fenmu = maxx - minn
        if fenmu==0:
            arr = np.zeros(arr.shape,np.uint8)
        else:
            arr = (((arr - minn) / fenmu) * 254).astype(np.uint8)
        return arr
    flows = norm(calTV1OpticalFlow(left,right))
    diff = norm(calRGBDifference(left,right))
    combine = np.zeros([flows.shape[0],flows.shape[1],3],np.uint8)
    combine[:,:,:2]=flows[...]
    combine[:,:,2]=np.ones(diff.shape,np.uint8)*128
    return combine

    print(flows.shape,diff.shape)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)  # 以网格的形式选取二维图像上等间隔的点，这里间隔为16，reshape成2行的array
    fx, fy = flow[y, x].T  # 取选定网格点坐标对应的光流位移
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)  # 将初始点和变化的点堆叠成2*2的数组
    lines = np.int32(lines + 0.5)  # 忽略微笑的假偏移，整数化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # 以初始点和终点划线表示光流运动
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # 在初始点（网格点处画圆点来表示初始点）
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi  # 得到运动的角度
    v = np.sqrt(fx * fx + fy * fy)  # 得到运动的位移长度
    hsv = np.zeros((h, w, 3), np.uint8)  # 初始化一个0值空3通道图像
    hsv[..., 0] = ang * (180 / np.pi / 2)  # B通道为角度信息表示色调
    hsv[..., 1] = 255  # G通道为255饱和度
    hsv[..., 2] = np.minimum(v * 4, 255)  # R通道为位移与255中较小值来表示亮度
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 将得到的HSV模型转换为BGR显示
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)  # 图像几何变换（线性插值），将原图像的像素映射到新的坐标上去
    return res


if __name__ == '__main__':


    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    cam = cv2.VideoCapture(r'E:\dataset\VIVA\01_01_01.avi')  # 读取视频
    #cam = cv2.VideoCapture(0)
    #ret, prev = cam.read()  # 读取视频第一帧作为光流输入的当前帧֡
    # prev = cv2.imread('E:\lena.jpg')

    # prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # show_hsv = False
    # show_glitch = False
    # cur_glitch = prev.copy()

    ret, frame1 = cam.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    #pFlow = cv2.DualTVL1OpticalFlow_create()
    while True:

            ret, frame2 = cam.read()
            if ret==False:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            #flow = np.zeros(shape=(next.shape[0],next.shape[1],2),dtype=np.float32)
            #flow = cv2.CreateMat(3, 3, cv2.CV_32FC2)
            #pFlow.calc(prvs,next,flow)
            combine = calOpticalFlowAndRGBDifference(prvs,next)
            prvs = next.copy()
            cv2.imshow('t',combine)
            cv2.waitKey(1)
            #flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15,  3, 5, 1.2, 0)
            #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #hsv[..., 0] = ang * 180 / np.pi / 2
            #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
