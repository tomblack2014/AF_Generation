import os

import numpy as np
from numpy.linalg import norm
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import struct
import math
import colorsys

def spilt_frames(avi_path, frame_dir, RESIZE_FACTOR):
    """
    将视频每帧分割并保存
    Args:
        avi_path: 视频路径
        frame_dir: 保存帧的文件夹位置
        RESIZE_FACTOR: 尺寸缩放比例

    Returns: 输入视频的帧数

    """
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    cap = cv.VideoCapture(avi_path)
    cnt_frame = 0
    while True:
        # success 表示是否成功，data是当前帧的图像数据；read读取一帧图像，移动到下一帧
        success, data = cap.read()
        if not success:
            break
        dim = (int(data.shape[1] * RESIZE_FACTOR), int(data.shape[0] * RESIZE_FACTOR))
        data = cv.resize(data, dim)
        im = Image.fromarray(data)
        im.save(frame_dir + '%06d' % cnt_frame + '.jpg')
        cnt_frame = cnt_frame + 1

    cap.release()

    return cnt_frame

def clip_over(arr):
    return np.clip(arr, 0, 359)

def field2img(field):
    h, w, n = field.shape
    max_idx = np.argmax(field, axis=2)
    max_idx = max_idx / (n - 1)  # 归一化到[0,1]范围内

    # 将h数组转换为float32类型，并将其扩展为3通道
    h = np.float32(max_idx)
    hsv = cv.merge([h * 360, np.ones_like(h), np.ones_like(h)])

    # 将HSV颜色空间转换为BGR颜色空间
    img_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB) * 255
    return img_bgr

def run():
    frame_dir = 'newdata/Frame/'    # 保存帧的文件夹位置
    frame_cnt = 500
    frame1 = cv.imread(os.path.join(frame_dir + '%06d' % 0 + '.jpg'))
    h = frame1.shape[0]  # 图像高度
    w = frame1.shape[1]  # 图像长度
    # mesh 网格点，表示x，y坐标
    mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))
    ori = np.zeros((h, w, 360), dtype=float)

    video_name = "my_video2024.mp4"
    frame_size = (w, h)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = 24
    video = cv.VideoWriter(video_name, fourcc, fps, frame_size)
    cnt = 0

    plt.ion()
    for cur_frame in range(frame_cnt - 1):
        # print('------------------------------------')
        print('Running frame ' + str(cur_frame))

        # main loop
        frame1 = cv.imread(os.path.join(frame_dir + '%06d' % (cur_frame * 20) + '.jpg'))
        frame2 = cv.imread(os.path.join(frame_dir + '%06d' % (cur_frame * 20 + 20) + '.jpg'))

        opt_hsv = np.zeros_like(frame1)  # 光流图
        opt_hsv[..., 1] = 255

        frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        opt_flow1 = cv.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        u = opt_flow1[..., 0].copy()  # x velocity
        v = opt_flow1[..., 1].copy()  # y velocity
        theta = np.floor(cv.phase(u, v) * (180 / math.pi))
        theta_clipped = clip_over(theta)

        for i in range(h):
            for j in range(w):
                ori[i, j, int(theta_clipped[i, j])] += 1

        im = np.uint8(field2img(ori))
        video.write(im)
        #filename = 'field/' + str(cur_frame) + '.jpg'
        #cv.imwrite(filename, im)

    video.release()

run()