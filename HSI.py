import cv2
import numpy as np

def RGB2HSI(rgb_img):
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]

    hsi_img = rgb_img.copy()

    B,G,R = cv2.split(rgb_img)
    #归一化到[0,1]
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))    #定义H通道
    I = (R + G + B) / 3.0       #计算I通道
    S = np.zeros((row,col))      #定义S通道
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #计算夹角
        h = np.zeros(col)               #定义临时数组
        #den>0且G>=B的元素h赋值为thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        #den>0且G<=B的元素h赋值为thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        #den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)
    #计算S通道
    for i in range(row):
        min = []
        #找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        #计算S通道
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        #I为0的值直接赋值0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    #扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img

def HSI2RGB(hsi_img):

    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]

    rgb_img = hsi_img.copy()

    H,S,I = cv2.split(hsi_img)
    #归一化到[0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H大于等于0小于120度时
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         #第一种情况
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H大于等于120度小于240度
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         #第二种情况
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        #H大于等于240度小于360度
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             #第三种情况
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img

def run_main():

    #读入图片
    rgb_img = cv2.imread('1.jpg',cv2.IMREAD_COLOR)
    #颜色空间转换
    hsi_img = RGB2HSI(rgb_img)
    rgb_img2 = HSI2RGB(hsi_img)
    #opencv库的颜色空间转换结果
    hsi_img2 = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HSV)
    rgb_img3 = cv2.cvtColor(hsi_img2,cv2.COLOR_HSV2BGR)
    cv2.imshow("Origin",rgb_img)
    cv2.imshow("HSI", hsi_img)
    cv2.imshow("RGB",rgb_img2)
    cv2.imshow("OpenCV_HSI",hsi_img2)
    cv2.imshow("OpenCV_RGB",rgb_img3)
    cv2.imwrite("HSI.jpeg",hsi_img)
    cv2.imwrite("RGB.jpeg", rgb_img2)
    cv2.imwrite("OpenCV_HSI.jpeg", hsi_img2)
    cv2.imwrite("OpenCV_RGB.jpeg", rgb_img3)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_main()