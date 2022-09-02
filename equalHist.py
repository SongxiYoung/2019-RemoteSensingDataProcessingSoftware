# -*-  coding: utf-8 -*-
from PIL import Image
from numpy import *
from pylab import *
from matplotlib import pyplot
# from scipy.misc import *
from scipy.misc import toimage
import cv2


# 定义直方图均衡化函数，这里传入的参数是灰度图像的数组和累积分布函数值
def histImageArr(im_arr, cdf):
    cdf_min = cdf[0]
    im_w = len(im_arr[0])
    im_h = len(im_arr)
    im_num = im_w * im_h
    color_list = []
    i = 0

    # 通过累积分布函数计算灰度转换值
    while i < 256:
        if i > len(cdf) - 1:
            color_list.append(color_list[i - 1])
            break
        tmp_v = (cdf[i] - cdf_min) * 255 / (im_num - cdf_min)
        color_list.append(tmp_v)
        i += 1

    # 产生均衡化后的图像数据
    arr_im_hist = []
    for itemL in im_arr:
        tmp_line = []
        for item_p in itemL:
            tmp_line.append(color_list[item_p])
        arr_im_hist.append(tmp_line)

    return arr_im_hist


# cdf是累积分布函数数值
def beautyImage(im_arr):
    imhist, bins = histogram(im_arr.flatten(), range(256))
    cdf = imhist.cumsum()

    return histImageArr(im_arr, cdf)

if __name__ == '__main__':
    im_source = Image.open('1.jpg')
    # 这一部分是对彩图的直方图均衡化例子
    arr_im_rgb = array(im_source)

    arr_im_rcolor, arr_im_gcolor, arr_im_bcolor = cv2.split(arr_im_rgb)
    '''
    arr_im_rcolor = []
    arr_im_gcolor = []
    arr_im_bcolor = []
    i = 0
    # 分离三原色通道
    for itemL in arr_im_rgb:
        arr_im_gcolor.append([])
        arr_im_rcolor.append([])
        arr_im_bcolor.append([])
        for itemC in itemL:
            arr_im_rcolor[i].append(itemC[0])
            arr_im_gcolor[i].append(itemC[1])
            arr_im_bcolor[i].append(itemC[2])
        i = 1 + i
    '''

    # 三个通道通过各自的分布函数来处理
    arr_im_rcolor_hist = beautyImage(array(arr_im_rcolor))
    arr_im_gcolor_hist = beautyImage(array(arr_im_gcolor))
    arr_im_bcolor_hist = beautyImage(array(arr_im_bcolor))

    # 合并三个通道颜色到图片
    i = 0
    arr_im_hist = []
    arr_im_rcolor_hist = array(arr_im_rcolor_hist).reshape(arr_im_rcolor.shape)
    arr_im_gcolor_hist = array(arr_im_gcolor_hist).reshape(arr_im_rcolor.shape)
    arr_im_bcolor_hist = array(arr_im_bcolor_hist).reshape(arr_im_rcolor.shape)
    arr_im_hist.append(arr_im_rcolor_hist)
    arr_im_hist.append(arr_im_gcolor_hist)
    arr_im_hist.append(arr_im_bcolor_hist)

    figure()
    im_beauty = toimage(array(arr_im_hist), 255)
    im_beauty.show()
