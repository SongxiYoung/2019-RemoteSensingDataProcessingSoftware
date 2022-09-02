# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 06:28:10 2018

@author: Lenovo
"""

from PIL import Image
import numpy
import random

def distance(p, c):
    s = sum(numpy.power((p-c), 2))  
    return numpy.sqrt(s)

def randinitialize(data, k):
    m, n = data.shape 
    center = numpy.zeros((k, n))
    index = [random.randint(0, m), random.randint(0, m), random.randint(0, m)]
    for i in range(k):
        center[i, :] = data[index[i], :]
    return center

def newcenter(x):
    new = numpy.mean(x)
    return new

def Kmeans(data, centers, k):
    m, p = centers.shape
    n = data.shape[0]
    result = numpy.empty((n,1))
    color = numpy.arange(0,k,1)
    classes = [[] for i in range(k)]
    new_centers = numpy.zeros((centers.shape))

    while new_centers.all() != centers.all():
        for i in range(n):
            a = numpy.zeros((1, k))
            for j in range(m):
                a[0][j] = distance(data[i,:], centers[j,:])
            dmax = numpy.min(a)
            for j in range(m):
                if dmax == a[0][j]:
                    classes[j].append(data[i,:])
                    result[i] = color[j]
                else:
                    pass
  
        for i in range(m):
            l = len(classes[i])
            x = numpy.array(classes[i]).reshape((l,3))
            new_centers[i] = newcenter(numpy.array(x[:,0]))

    return new_centers, classes, result

def accuracy(classes):
    m = len(classes)
    for i in range(m):
        print("class", i+1)
        print(numpy.array(classes[i]))

if __name__ == '__main__':
    im = Image.open("M.jpg")
    #im.show()
    img_arr = numpy.array(im) 
    m = img_arr.shape[0]*img_arr.shape[1]
    n = img_arr.shape[2]
    data = img_arr.reshape(m, n)
    centers = randinitialize(data, 3)      #3是预估类别数
    
    new, classes, result = Kmeans(data, centers, 3)
    print("New centers:")
    print(new)
    #accuracy(classes)
    #print(result)
    
    k = 0
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if result[k]==0:
                img_arr[i, j, :] = [255, 0, 0]
            if result[k]==1:
                img_arr[i, j, :] = [255, 250, 250]
            if result[k]==2:
                img_arr[i, j, :] = [255, 255, 0]
            if result[k]==3:
                img_arr[i, j, :] = [255, 192, 203]
            else:
                pass
            k = k + 1
    
    new_img = Image.fromarray(img_arr) 
    new_img.show()  
    new_img.save("Kmeans_test.jpg")
            
            