# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:30:12 2018

@author: Lenovo
"""

import numpy
from PIL import Image

def distance(p, c):
    p = numpy.array(p)
    c = numpy.array(c)
    return sum(numpy.power((p - c), 2))


def randinitialize(data, k):
    m, n = data.shape
    center = numpy.zeros((k, n))
    for i in range(n):
        dmin, dmax = numpy.min(data[:, i]), numpy.max(data[:, i])
        center[:, i] = dmin + (dmax - dmin) * numpy.random.rand(k)
    return list(center)


def newcenter(x):
    n = x.shape[1]
    new = []
    for i in range(n):
        new.append(numpy.mean(x[:, i]))
    return new


def classify(result, data, centers, k0):
    m = len(centers)
    n = data.shape[0]
    test_color = numpy.arange(0, k0, 1)
    classes = [[] for i in range(k0)]

    for i in range(n):
        a = numpy.zeros((1, k0))
        for j in range(m):
            a[0][j] = distance(data[i, :], centers[j])
        dmin = numpy.min(a)
        for j in range(m):
            if dmin == a[0][j]:
                classes[j].append(data[i, :])
                result.append(test_color[j])
    return classes, result


def Std(data, centers):
    m, n = data.shape
    D = numpy.zeros((1, n))
    for j in range(n):
        sum_square = 0
        for i in range(m):
            sum_square += numpy.sum(pow((data[i, :] - centers), 2))
        D[0][j] = sum_square / m
    return numpy.max(D)


def spiltClasses(result, new_centers, classes, sigma, n_min, k):
    var = []
    final_class = classes
    final_center = new_centers
    m = len(classes)
    for i in range(m):
        var.append(Std(numpy.array(classes[i]), final_center[i]))  # 计算每个聚类中样本距离的标准差向量最大分量
    for i in range(len(final_class)):
        if var[i] > sigma and numpy.array(final_class[i]).shape[0] >= 2 * n_min:
            p = []
            k = k + 1
            p1 = numpy.add(final_center[i], var[i])
            p2 = numpy.array(final_center[i]) - var[i]
            p.append(p1)
            p.append(p2)
            c, result = classify(result, numpy.array(final_class[i]), p, 2)
            del final_center[i]
            del final_class[i]
            final_center.append(p1)
            final_center.append(p2)
            final_class.append(c)
        else:
            pass

    return final_center, final_class, k, result


def mergeClasses(new_centers, classes, d_min, k):
    D = []
    final_class = classes
    final_center = new_centers
    for i in range(len(final_center)):
        for j in range(len(final_center)):
            D.append(distance(final_center[i], final_center[j]))
    for i in range(len(final_center)):
        for j in range(len(final_center)):
            if i != j and D[len(final_center) * i + j] < d_min:
                low = min(i, j)
                up = max(i, j)
                n1 = len(final_class[i])
                n2 = len(final_class[j])
                mid = numpy.array((final_center[i] * n1 + final_center[j] * n2)) * (1 / (n1 + n2))
                final_center[low] = list(mid)
                del final_center[up]
                final_class[low] = final_class[low] + final_class[up]
                del final_class[up]
                k = k - 1
                for a in range(len(final_center) + 1):
                    del D[a * len(final_center) + j]
                for b in range(i * (len(final_center) - 1), (i + 1) * (len(final_center) - 1) + 1):
                    del D[b]
            break
        break

    return final_center, final_class, k


def ISODATA(data, centers, k0, d_min, sigma, n_min, iteration):
    k = k0
    new_centers = centers
    classes = []
    result = []

    for e in range(1, iteration + 1):
        print(e, "times")
        print("current k:", k)
        classes, result = classify(result, data, new_centers, k)  # 样本分给最近的聚类

        for i in range(len(classes)):
            if i < len(classes) and len(classes[i]) < n_min:  # 样本数目过少则取消该聚类
                k = k - 1
                del new_centers[i]
                classes, result = classify(result, data, new_centers, k)
                print("merge.")
            else:
                pass

        for i in range(len(new_centers)):  # 修正各聚类中心的值
            if 0 == len(classes[i]):
                k = k - 1
                del classes[i]
                del new_centers[i]
            else:
                x = numpy.array(classes[i])
                new_centers[i] = newcenter(x)
        if k <= k0 / 2:
            new_centers, classes, k, result = spiltClasses(result, new_centers, classes, sigma, n_min, k)  # 分裂
            print("spiltClasses called!")
        elif k >= 2 * k0:
            new_centers, classes, k = mergeClasses(new_centers, classes, d_min, k)  # 合并
            print("mergeClasses called!")
        else:
            pass

    return new_centers, classes, k, result

def paint(result, img_arr):
    k = 0
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if result[k] == 0:
                img_arr[i, j, :] = [255, 0, 0]
            if result[k] == 2:
                img_arr[i, j, :] = [255, 255, 0]
            if result[k] == 1:
                img_arr[i, j, :] = [255, 192, 203]
            if result[k] == 3:
                img_arr[i, j, :] = [160, 32, 240]
            k = k + 1

    new_img = Image.fromarray(img_arr)
    #new_img.show()
    new_img.save("ISODATA_test2.jpg")
    return new_img

if __name__ == '__main__':
    im = Image.open("hist.jpg")
    print("图像读取成功")
    img_arr = numpy.array(im)
    m = img_arr.shape[0] * img_arr.shape[1]
    n = img_arr.shape[2]
    X = img_arr.reshape(m, n)
    #result = []

    k0 = 3  # 预期聚类中心数目
    d_min = 1  # 小于此距离两个聚类合并
    sigma = 50  # 样本中距离分布的标准差
    n_min = 10  # 少于此数就不作为一个独立的聚类
    iteration = 2  # 迭代次数

    centers = randinitialize(X, k0)
    print("正在执行ISODATA算法，需要耗费一些时间，如不想等待可直接查看目录中的分类结果")
    new, classes, k, result = ISODATA(X, centers, k0, d_min, sigma, n_min, iteration)
    print("New centers:")
    print(new)
    print("number of classes:", k)

    paint(result, img_arr)