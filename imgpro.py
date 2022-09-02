import cv2
import math
import numpy as np

class Img:
    def __init__(self,image,rows,cols,center=[0,0]):
        self.src=image #原始图像
        self.rows=rows #原始图像的行
        self.cols=cols #原始图像的列
        self.center=center #旋转中心，默认是[0,0]
    '''
    def Move(self,delta_x,delta_y):      #平移
        #delta_x>0左移，delta_x<0右移
        #delta_y>0上移，delta_y<0下移
        self.transform=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])

    def Zoom(self,factor):               #缩放
        #factor>1表示缩小；factor<1表示放大
        self.transform=np.array([[factor,0,0],[0,factor,0],[0,0,1]])
    '''
    def Rotate(self,beta):               #旋转
        #beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform=np.array([[math.cos(beta),-math.sin(beta),0],
                                 [math.sin(beta), math.cos(beta),0],
                                 [    0,              0,         1]])

    def Process(self):
        self.dst=np.zeros((self.rows,self.cols, self.src.shape[2]),dtype=np.uint8)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos=np.array([i-self.center[0],j-self.center[1],1])
                [x,y,z]=np.dot(self.transform,src_pos)
                x=int(x)+self.center[0]
                y=int(y)+self.center[1]

                if x >= self.src.shape[0] or y >= self.src.shape[1] or x < 0 or y < 0:
                    self.dst[i][j][:]=0
                else:
                    self.dst[i][j][:]=self.src[x][y][:]

if __name__=='__main__':
    src=cv2.imread('1.jpg',0)
    rows = src.shape[0]
    cols = src.shape[1]

    src=Img(src,rows,cols, [int(rows/2), int(cols/2)])
    src.Rotate(math.radians(80)) #旋转
    src.Process()
    #cv2.imshow('dst', img.dst)
    cv2.imwrite("1.png", src.dst)
    cv2.waitKey(0)