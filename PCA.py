'''
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示按照列来求均值，如果输入list,则axis=1


def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print
        "k must lower than feature number"
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里需要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=-1)).astype(np.float)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1');
    plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()


# 根据数据集data.txt
def main():
    im = cv2.imread("1.jpg")
    img = im.reshape((im.shape[0] * im.shape[1], im.shape[2]))
    k = 3
    return pca(img, k)


if __name__ == "__main__":
    finalData, reconMat = main()
    print(finalData)
'''

import numpy
import cv2
import matplotlib.pyplot as plt  

def load(filename):
    file = open(filename)
    data = []
    labels = []
    for eachline in file:
        linedata = eachline.strip().split(",")
        data.append([float(linedata[0]), float(linedata[1]), float(linedata[2]),
                     float(linedata[3])])
    data = numpy.array(data)
    return data

def mean(data):
    meanval = numpy.mean(data,axis=0)     
    newdata = data-meanval
    return newdata, meanval

def pca(data, n):
    newdata, meanval = mean(data)
    cov = numpy.cov(newdata, rowvar=0)
    eigVals,eigVects=numpy.linalg.eig(numpy.mat(cov))
    valsort = numpy.argsort(eigVals)
    index = valsort[-1:-(n+1):-1]
    newVec = eigVects[:,index]
    lowdata = numpy.dot(newdata, newVec)
    return lowdata

if __name__ == '__main__':
    im = cv2.imread("1.jpg")
    img = im.reshape((im.shape[0]*im.shape[1], im.shape[2]))
    lowdata = pca(img, 3)
    print(lowdata)
    #x = list(lowdata[:,0])
    #y = list(lowdata[:,1])
    #plt.figure(figsize=(9,6))
    #plt.scatter(x,y)
    #plt.show()

