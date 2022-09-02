import sys
import numpy as np
import math
import cv2
import imgpro  #旋转缩放
from PIL import Image
import equalHist as eh
import window
import convolve as co
import PCA
import HSI
import compute
from pylab import *
from scipy.misc import toimage
import Classification_ISO as CISO
import Classification_K as CK
import NaiveBayesMulti as NBM
from scipy import ndimage
import matplotlib.pyplot as plt
from osgeo import gdalnumeric
from osgeo import gdal
# 必要的引用。基本控件位于pyqt5.qtwidgets模块中。
from PyQt5.QtCore import Qt,pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication,QWidget,QToolTip,QPushButton,QMessageBox,QDesktopWidget, QLabel,\
    QHBoxLayout, QVBoxLayout,QGridLayout, QLineEdit, QTextEdit,QMainWindow, QAction, qApp, QLCDNumber, \
    QSlider, QInputDialog,QFrame, QColorDialog,QFileDialog,QScrollArea
from PyQt5.QtGui import QIcon,QFont,QColor,QPixmap,QImage
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtCore import QCoreApplication
num = 0
ban = []
basep = []

class RSImgPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.img = None
        self.scale=0
        self.initUI()  # 界面绘制交给InitUi方法

    def initUI(self):
        #状态栏
        self.statusBar()
        # 子窗口
        w = QWidget()
        w.resize(250, 150)
        #子窗口的布局
        grid = QGridLayout()
        self.label=QLabel('RS image')
        self.label.resize(450,450)
        grid.addWidget(self.label,0,0)
        w.setLayout(grid)
        menubar = self.menuBar()

        self.setCentralWidget(w)
        self.topFiller = QWidget()
        self.topFiller.setMinimumSize(250, 2000)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.topFiller)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.scroll)
        w.setLayout(self.vbox)

        self.another = window.Window()#wrap window

        # 添加菜单
        self.subwidget=w
        self.setCentralWidget(self.subwidget)
        fileMenu = menubar.addMenu('&File')
        DisplayMenu = menubar.addMenu('&Display')
        filterMenu = menubar.addMenu('&Filter')
        classfiMenu = menubar.addMenu('&Classification')
        transformMenu = menubar.addMenu('&Transform')
        RegistMenu = menubar.addMenu('&Registration')


        # 添加事件
        openfile_action = self.open_action()               #file
        fileMenu.addAction(openfile_action)
        fileMenu.addAction(self.save_action())

        grayorrgb = self.grayorrgb_action()                #display
        DisplayMenu.addAction(grayorrgb)
        stretch_action = self.linearstretch_action()
        DisplayMenu.addAction(stretch_action)
        sta_action = self.statistics_action()
        DisplayMenu.addAction(sta_action)
        DisplayMenu.addAction(self.downsample_action())
        DisplayMenu.addAction(self.upsample_action())
        DisplayMenu.addAction(self.histequal_action())
        DisplayMenu.addAction(self.imagefuse_action())

        filterMenu.addAction(self.HPF_action())            #filter
        filterMenu.addAction(self.sobel_action())
        filterMenu.addAction(self.Roberts_action())
        filterMenu.addAction(self.userdefined_action())

        transformMenu.addAction(self.PCA_action())        #PCA

        RegistMenu.addAction(self.sGCPs_action())         #Registration
        RegistMenu.addAction(self.rGCPs_action())

        classfiMenu.addAction(self.iso_action())            #classification
        classfiMenu.addAction(self.Kmeans_action())
        classfiMenu.addAction(self.bayes_action())

        self.setGeometry(300, 300,300,300)
        self.setWindowTitle('RSImgPro')
        self.setWindowIcon(QIcon('1.jpg'))
        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def openfileDialog(self):                                                                                           #open image
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            img = gdalnumeric.LoadFile(fname[0])                                     #6*400*640
            #self.dataset = gdal.Open(fname[0])
            #img = np.array(img)
            if len(img.shape)==3:
                self.all = np.transpose(img, [1, 2, 0]).astype(img.dtype)
                global num
                num = img.shape[0]
                self.img = np.transpose(img, [1, 2, 0]).astype(img.dtype)                #400*640*6
                self.choosebands()
            else:
                self.img = img
                self.all = self.img
                qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                              QImage.Format_Grayscale8)
                self.pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(self.pixmap)

    def choosebands(self):
        w = QWidget()
        w.setWindowTitle('image fuse')
        w.resize(250, 150)
        grid = QVBoxLayout()

        global num
        items = []
        for i in range(num):
            items.append((i, '%d band' % (i + 1)))
        for id_, txt in items:
            self.checkBox = QtWidgets.QCheckBox(txt, self)
            self.checkBox.id_ = id_
            self.checkBox.stateChanged.connect(self.checkLanguage)  # 1
            grid.addWidget(self.checkBox)
        self.lMessage = QtWidgets.QLabel(self)

        button1 = QPushButton('confirm')
        button1.clicked.connect(self.loadbands)
        grid.addWidget(button1)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def checkLanguage(self, state):
        checkBox = self.sender()
        if state == QtCore.Qt.Unchecked:
            self.lMessage.setText(u'取消选择了{0}: {1}'.format(checkBox.id_, checkBox.text()))
            ban.remove(checkBox.id_)
        elif state == QtCore.Qt.Checked:
            self.lMessage.setText(u'选择了{0}: {1}'.format(checkBox.id_, checkBox.text()))
            ban.append(checkBox.id_)

    def loadbands(self):
        global ban
        if len(ban) == 3:
            r, g, b = ban
            R = self.img[:, :, r]
            G = self.img[:, :, g]
            B = self.img[:, :, b]
            self.img = cv2.merge([R, G, B])
            qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                          QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(self.pixmap)
            self.rgb = np.array(self.img)
        elif len(ban) == 1:
            gray = ban[0]
            self.img = self.img[:, :, gray]
            qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                          QImage.Format_Grayscale8)
            self.pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(self.pixmap)
        else:
            QMessageBox.information(self, "error", "请选择单波段或三波段",
                                    QMessageBox.Yes)
        ban = []

    def open_action(self):
        exitAction = QAction('&open', self)
        exitAction.setStatusTip('open')
        exitAction.triggered.connect(self.openfileDialog)
        return exitAction

    def save_action(self):                                                                                              #save image
        exitAction = QAction('&save', self)
        exitAction.triggered.connect(self.saveimg)
        return exitAction

    def saveimg(self):
        if (self.img.all != None):
            fname = QFileDialog.getSaveFileName()
            if fname[0]:
                img2=cv2.cvtColor(self.img,cv2.COLOR_RGB2BGR)                  #opencv BGR
                cv2.imwrite(fname[0],img2)
        return

    def grayorrgb_action(self):                                                                                         #grayorrgb
        exitAction = QAction('&gray or rgb', self)
        exitAction.triggered.connect(self.grayorrgb)
        return exitAction

    def grayorrgb(self):
        if (self.img.all != None):
            print('converting...')
            if (len(self.img.shape) == 3):
                height = self.img.shape[0]
                width = self.img.shape[1]
                channels = self.img.shape[2]

                value = [0] * 3
                gray_img = np.zeros([height, width], np.uint8)
                for row in range(height):
                    for column in range(width):
                        for chan in range(channels):
                            value[chan] = self.img[row, column, chan]
                        R = value[2]
                        G = value[1]
                        B = value[0]
                        new_value = 0.2989 * R + 0.5870 * G + 0.1140 * B  # 转为灰度像素
                        gray_img[row, column] = new_value

                self.img = gray_img
                qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                              QImage.Format_Grayscale8)
                self.pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(self.pixmap)

            else:
                self.img = self.rgb
                qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                              QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(self.pixmap)
        return

    def statistics_action(self):
        exitAction = QAction('&statistics', self)
        exitAction.triggered.connect(self.statistics)
        return exitAction

    def statistics(self):
        img = np.array(self.img)

        maxMat = np.zeros([1, img.shape[2]], np.uint8)
        minMat = np.zeros([1, img.shape[2]], np.uint8)
        meanMat = np.zeros([1, img.shape[2]], np.uint8)
        varMat = np.zeros([1, img.shape[2]], np.uint8)
        stdMat = np.zeros([1, img.shape[2]], np.uint8)
        for i in range(img.shape[2]):
            maxPix = np.max(img[:, :, i])  # max
            minPix = np.min(img[:, :, i])  # min
            meanPix = np.mean(img[:, :, i])  # mean
            x = img.flatten()
            var = sum(pow((x - meanPix), 2)) / (img.shape[0] * img.shape[1]) - 1
            stdev = np.sqrt(var)  # stdev
            maxMat[0, i] = maxPix
            minMat[0, i] = minPix
            meanMat[0, i] = meanPix
            stdMat[0, i] = stdev
            varMat[0, i] = var

        f = open('st.txt', 'w')
        f.write('R   G   B\n')
        f.write('max')
        f.write(str(maxMat))
        f.write('\n')
        f.write('min')
        f.write(str(minMat))
        f.write('\n')
        f.write('mean')
        f.write(str(meanMat))
        f.write('\n')
        f.write('var')
        f.write(str(varMat))
        f.close()

        x = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        covMat = np.zeros([img.shape[2], img.shape[2]])
        corMat = np.zeros([img.shape[2], img.shape[2]])
        for i in range(img.shape[2]):
            for j in range(img.shape[2]):
                sp1 = sum(np.multiply(x[:, i], x[:, j]))
                sp2 = sum(x[:, i]) * sum(x[:, j]) / (x.shape[0] - 1)
                cov = (sp1 - sp2) / (x.shape[0] - 1)
                covMat[i][j] = cov
                cor = cov / (stdMat[0, j] * stdMat[0, i])
                corMat[i][j] = cor
        print('ok')
        color = ['red', 'green', 'blue', 'black', 'white', 'yellow']
        plt.figure("red")
        plt.hist(x[:, 0], bins=256, normed=1, facecolor=color[0], alpha=0.75)
        plt.savefig('red.png')
        plt.figure("green")
        plt.hist(x[:, 1], bins=256, normed=1, facecolor=color[1], alpha=0.75)
        plt.savefig('green.png')
        plt.figure("blue")
        plt.hist(x[:, 2], bins=256, normed=1, facecolor=color[2], alpha=0.75)
        plt.savefig('blue.png')
        plt.figure("all")
        plt.hist(x[:, 0], bins=256, normed=1, facecolor=color[0], alpha=0.75)
        plt.hist(x[:, 1], bins=256, normed=1, facecolor=color[1], alpha=0.75)
        plt.hist(x[:, 2], bins=256, normed=1, facecolor=color[2], alpha=0.75)
        plt.savefig('all.png')
        self.stashow()
        return

    def stashow(self):
        w = QWidget()
        w.setWindowTitle('statistics')
        w.resize(250, 150)

        hbox = QHBoxLayout()  # 水平布局
        Button1 = QPushButton("R-red")
        Button2 = QPushButton("G-green")
        Button3 = QPushButton("B-blue")
        #Button4 = QPushButton("all")
        hbox.addWidget(Button1)
        hbox.addWidget(Button2)
        hbox.addWidget(Button3)
        #hbox.addWidget(Button4)

        grid = QVBoxLayout()
        hist = QLabel('histogram')
        hist.resize(750, 750)
        png = QPixmap('all.png')
        hist.setPixmap(png)
        txt = QLabel('statistics')
        file = open('sta.txt')
        file_text = file.read()
        txt.setText(file_text)
        grid.addLayout(hbox)
        grid.addWidget(hist)
        grid.addWidget(txt)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def HPF_action(self):
        exitAction = QAction('&HPF', self)
        exitAction.triggered.connect(self.HPF)
        return exitAction

    def HPF(self):
        print('filtering...')
        kernel_3x3 = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])

        kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                               [-1, 1, 2, 1, -1],
                               [-1, 2, 4, 2, -1],
                               [-1, 1, 2, 1, -1],
                               [-1, -1, -1, -1, -1]])
        self.img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        k3 = ndimage.filters.convolve(self.img, kernel_3x3, mode='constant', cval=0.0)
        #k5 = ndimage.convolve(self.img, kernel_5x5, mode='constant', cval=0.0)
        self.img = k3
        qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                      QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        return

    def sobel_action(self):
        exitAction = QAction('&sobel', self)
        exitAction.triggered.connect(self.sobel)
        return exitAction

    def sobel(self):
        Gxf = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

        Gyf = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

        self.img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        kx = ndimage.filters.convolve(self.img, Gxf, mode='constant', cval=0.0)
        ky = ndimage.convolve(self.img, Gyf, mode='constant', cval=0.0)
        self.img = kx + ky
        qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                      QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        return

    def Roberts_action(self):
        exitAction = QAction('&Roberts', self)
        exitAction.triggered.connect(self.Roberts)
        return exitAction

    def Roberts(self):
        rob = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
        rob2 = np.array([[0, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        res = co.convolve(self.img, rob)
        res2 = co.convolve(self.img, rob2)
        self.img = res + res2
        qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                      QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        return

    def userdefined_action(self):
        exitAction = QAction('&user_defined', self)
        exitAction.triggered.connect(self.define)
        return exitAction

    def define(self):
        w = QWidget()
        w.setWindowTitle('filter')
        w.resize(250, 150)

        hbox1 = QHBoxLayout()  # 水平布局
        self.t1 = QLineEdit()
        self.t2 = QLineEdit()
        self.t3 = QLineEdit()
        hbox1.addWidget(self.t1)
        hbox1.addWidget(self.t2)
        hbox1.addWidget(self.t3)

        hbox2 = QHBoxLayout()  # 水平布局
        self.t4 = QLineEdit()
        self.t5 = QLineEdit()
        self.t6 = QLineEdit()
        hbox2.addWidget(self.t4)
        hbox2.addWidget(self.t5)
        hbox2.addWidget(self.t6)

        hbox3 = QHBoxLayout()  # 水平布局
        self.t7 = QLineEdit()
        self.t8 = QLineEdit()
        self.t9 = QLineEdit()
        hbox3.addWidget(self.t7)
        hbox3.addWidget(self.t8)
        hbox3.addWidget(self.t9)

        hbox4 = QHBoxLayout()  # 水平布局
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.filter)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        hbox4.addWidget(button3)
        hbox4.addWidget(button4)

        grid = QVBoxLayout()
        grid.addLayout(hbox1)
        grid.addLayout(hbox2)
        grid.addLayout(hbox3)
        grid.addLayout(hbox4)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def filter(self):
        fl = np.zeros((3, 3))
        fl[0][0] = float(self.t1.text())
        fl[0][1] = float(self.t2.text())
        fl[0][2] = float(self.t3.text())
        fl[1][0] = float(self.t4.text())
        fl[1][1] = float(self.t5.text())
        fl[1][2] = float(self.t6.text())
        fl[2][0] = float(self.t7.text())
        fl[2][1] = float(self.t8.text())
        fl[2][2] = float(self.t9.text())
        res = co.convolve(self.img, fl)
        self.img = res
        qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                      QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)

    def linearstretch_action(self):                                                                                     #linearstretch
        exitAction = QAction('&linear stretch', self)
        exitAction.setStatusTip('to linear stretch image')
        exitAction.triggered.connect(self.linearstretch)
        return exitAction

    def linearstretch(self):
        if(self.img.all != None):
            min = np.min(self.img)
            max = np.max(self.img)
            self.img = (self.img / (max - min) * 255).astype(np.uint8)
            qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(self.pixmap)
        return

    def upsample_action(self):
        exitAction = QAction('&sacle up', self)
        exitAction.triggered.connect(self.upsample)
        return exitAction

    def upsample(self):
        if (self.img.all != None):
            if(self.scale !=2):
                self.img = cv2.resize(self.img,(int(self.img.shape[1] * 2),int(self.img.shape[0] * 2)))

                qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                              QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(self.pixmap)
                self.scale=self.scale + 1

        return

    def downsample_action(self):
        exitAction = QAction('&sacle down', self)
        exitAction.triggered.connect(self.downsample)
        return exitAction

    def downsample(self):
        if (self.img.all != None):
            if(self.scale != -2):
                self.img = cv2.resize(self.img,(int(self.img.shape[1]/2),int(self.img.shape[0]/2)))
                qimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                              QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(self.pixmap)
                self.scale=self.scale - 1
        return

    def histequal_action(self):
        exitAction = QAction('&equalize hist', self)
        exitAction.triggered.connect(self.histequal)
        return exitAction

    def imagefuse_action(self):
        exitAction = QAction('&Rs fuse', self)
        exitAction.triggered.connect(self.imagefuse)
        return exitAction

    def imagefuse(self):
        w = QWidget()
        w.setWindowTitle('image fuse')
        w.resize(250, 150)
        # 子窗口的布局
        grid = QGridLayout()
        button1 = QPushButton('choose multispectral image')
        button1.clicked.connect(self.choosemsp)
        button2 = QPushButton('choose pan image')
        button2.clicked.connect(self.choosepan)
        button_save = QPushButton('choose save path')
        button_save.clicked.connect(self.choosesavepath)
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.img_fuse)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        self.text1=QLineEdit()
        self.text2=QLineEdit()
        self.text3 = QLineEdit()
        grid.addWidget(button1, 0, 0)
        grid.addWidget(button2, 1, 0)
        grid.addWidget(button_save, 2, 0)
        grid.addWidget(self.text1, 0, 1)
        grid.addWidget(self.text2, 1, 1)
        grid.addWidget(self.text3, 2, 1)
        grid.addWidget(button3, 3, 0)
        grid.addWidget(button4, 3, 1)
        w.setLayout(grid)
        self.w=w
        self.w.show()
        return

    def img_fuse(self):
        img_RGB = gdalnumeric.LoadFile(self.msp)
        img_RGB = img_RGB[:3, :, :]
        min = np.min(img_RGB)
        max = np.max(img_RGB)
        if (max > 255 or min < 0):
            img_RGB = (img_RGB / (max - min) * 255).astype(np.uint8)
        img_RGB = np.transpose(img_RGB, [1, 2, 0])
        img_pan = gdalnumeric.LoadFile(self.pan)
        min_pan = np.min(img_pan)
        max_pan = np.max(img_pan)
        if (max_pan > 255 or min_pan < 0):
            img_pan = (img_pan / (max_pan - min_pan) * 255).astype(np.uint8)
        img_RGB = cv2.resize(img_RGB,(img_pan.shape[1], img_pan.shape[0]))
        hls = HSI.RGB2HSI(img_RGB)
        #hls = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
        H, L, S = cv2.split(hls)
        mean_pan = np.mean(img_pan)
        std_pan = np.std(img_pan)
        mean_l = np.mean(L)
        std_l = np.std(L)
        img_pan = ((img_pan - mean_pan) / std_pan * std_l + mean_l).astype(np.uint8)
        img_fused_hls = cv2.merge([H, img_pan, S])
        #img_fused = cv2.cvtColor(img_fused_hls, cv2.COLOR_HSV2BGR)
        img_fused = HSI.HSI2RGB(img_fused_hls)
        cv2.imwrite(self.savepath, img_fused)
        self.w.close()
        QMessageBox.information(self, "Information",
                                self.tr("融合完成!"))
        return

    def choosemsp(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            self.text1.setText(fname[0])
            self.msp=fname[0]
        return

    def choosepan(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            self.text2.setText(fname[0])
            self.pan = fname[0]
        return

    def choosesavepath(self):
        fname = QFileDialog.getSaveFileName()
        if fname[0]:
            self.text3.setText(fname[0])
            self.savepath = fname[0]
        return

    def histequal(self):
        if (self.img.all != None):
            R,G,B=cv2.split(self.img)

            # 三个通道通过各自的分布函数来处理
            arr_im_rcolor_hist = eh.beautyImage(np.array(R))
            arr_im_gcolor_hist = eh.beautyImage(np.array(G))
            arr_im_bcolor_hist = eh.beautyImage(np.array(B))
            # 合并三个通道颜色到图片
            arr_im_hist = []
            arr_im_rcolor_hist = np.array(arr_im_rcolor_hist).reshape(R.shape)
            arr_im_gcolor_hist = np.array(arr_im_gcolor_hist).reshape(G.shape)
            arr_im_bcolor_hist = np.array(arr_im_bcolor_hist).reshape(B.shape)
            arr_im_hist.append(arr_im_rcolor_hist)
            arr_im_hist.append(arr_im_gcolor_hist)
            arr_im_hist.append(arr_im_bcolor_hist)

            figure()
            im_beauty = toimage(np.array(arr_im_hist), 255)
            im_beauty.save("hist.jpg")
            '''
            R=cv2.equalizeHist(R)
            G=cv2.equalizeHist(G)
            B=cv2.equalizeHist(B)
            '''
            self.img = cv2.imread("hist.jpg")
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1] * 3,
                          QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(self.pixmap)
        return

    def PCA_action(self):
        exitAction = QAction('&PCA', self)
        exitAction.triggered.connect(self.PCAbands)
        return exitAction

    def PCAbands(self):
        w = QWidget()
        w.setWindowTitle('pca')
        w.resize(250, 150)

        hbox1 = QHBoxLayout()  # 水平布局
        self.pcanum = QLineEdit()
        self.pcanum.setPlaceholderText("input PCA后维数")
        hbox1.addWidget(self.pcanum)
        hbox4 = QHBoxLayout()  # 水平布局
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.pca)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        hbox4.addWidget(button3)
        hbox4.addWidget(button4)
        grid = QVBoxLayout()
        grid.addLayout(hbox1)
        grid.addLayout(hbox4)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def pca(self):
        k = int(self.pcanum.text())
        img_arr = self.all
        data = img_arr.reshape(img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2])
        lowdata = PCA.pca(data, k)
        self.img = (lowdata / (np.max(lowdata) - np.min(lowdata)) * 255).astype(np.uint8)
        qimg = QImage(self.img, img_arr.shape[1], img_arr.shape[0], img_arr.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        return

    def sGCPs_action(self):
        exitAction = QAction('&select GCPs', self)
        exitAction.triggered.connect(self.selectGCPs)
        return exitAction

    def selectGCPs(self):
        w = QWidget()
        w.setWindowTitle('select image')
        w.resize(250, 150)
        # 子窗口的布局
        grid = QGridLayout()
        button2 = QPushButton('choose wrap image')
        button2.clicked.connect(self.choosewrap)
        button_save = QPushButton('choose save path')
        button_save.clicked.connect(self.savepath)
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.manual)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        self.text3 = QLineEdit()
        grid.addWidget(button2, 2, 0)
        grid.addWidget(button_save, 1, 0)
        grid.addWidget(self.text3, 1, 1)
        grid.addWidget(button3, 3, 0)
        grid.addWidget(button4, 3, 1)
        w.setLayout(grid)
        self.w = w
        self.w.show()
        return

    def choosewrap(self):
        self.another.show()
        return

    def savepath(self):
        fname = QFileDialog.getSaveFileName()
        if fname[0]:
            self.chosave.setText(fname[0])
            self.save2 = fname[0]
        return

    #spot->base->geo->mainwin   tm->wrap->nogeo
    def manual(self):
        global basep
        print(self.another.wrapp)
        pass

    def mousePressEvent(self, QMouseEvent):
        print(QMouseEvent.pos())
        global basep
        basep.append(QMouseEvent.pos())

    def rGCPs_action(self):
        exitAction = QAction('&read GCPs', self)
        exitAction.triggered.connect(self.readGCPs)
        return exitAction

    def readGCPs(self):
        w = QWidget()
        w.setWindowTitle('select image')
        w.resize(250, 150)
        # 子窗口的布局
        grid = QGridLayout()

        button1 = QPushButton('choose base image')
        button1.clicked.connect(self.chobase)
        button2 = QPushButton('choose wrap image')
        button2.clicked.connect(self.chowrap)
        button_save = QPushButton('choose save path')
        button_save.clicked.connect(self.choosesavepath)
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.auto)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        button5 = QPushButton('choose GCPs file')
        button5.clicked.connect(self.chooseGCPs)
        self.text1 = QLineEdit()
        self.text2 = QLineEdit()
        self.text3 = QLineEdit()
        self.text4 = QLineEdit()
        grid.addWidget(button1, 0, 0)
        grid.addWidget(button2, 1, 0)
        grid.addWidget(button5, 2, 0)
        grid.addWidget(button_save, 3, 0)
        grid.addWidget(self.text1, 0, 1)
        grid.addWidget(self.text2, 1, 1)
        grid.addWidget(self.text3, 3, 1)
        grid.addWidget(self.text4, 2, 1)
        grid.addWidget(button3, 4, 0)
        grid.addWidget(button4, 4, 1)
        w.setLayout(grid)
        self.w = w
        self.w.show()
        return

    def chobase(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            self.text1.setText(fname[0])
            self.base = fname[0]
        return

    def chowrap(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            self.text2.setText(fname[0])
            self.wrap = fname[0]
        return

    def chooseGCPs(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        if fname[0]:
            self.text4.setText(fname[0])
            self.GCP = fname[0]
        return

    def auto(self):
        self.baseimg = gdalnumeric.LoadFile(self.base)
        self.wrapimg = gdalnumeric.LoadFile(self.wrap)
        dataSet = []
        bas = []
        wra = []
        with open(self.GCP) as fr:
            for line in fr.readlines()[5:8]:
                curline = line.strip()
                test = curline.split()
                x1, y1 = float(test[0]), float(test[1])
                x2, y2 = float(test[2]), float(test[3])
                bas.append((x1, y1))
                wra.append((x2, y2))
                dataSet.append(test)
        wrap = np.ones((3, 3))
        base = np.ones((3, 2))
        for i in range(3):
            wrap[i][1], wrap[i][2] = wra[i]
            base[i][0], base[i][1] = bas[i]

        a1 = np.dot(wrap.transpose(), wrap)
        B = np.linalg.inv(a1)
        coeff = np.dot(np.dot(B, wrap.transpose()), base)  # wrap:  (6, 498, 414) to  base:  (1390, 1071)
        # print(coeff)

        a, b = self.wrapimg.shape[1], self.wrapimg.shape[2]
        test = np.zeros((4, 3))
        test[0][0] = 1
        test[1][0] = 1
        test[2][0] = 1
        test[3][0] = 1
        test[1][2] = b
        test[2][1] = a
        test[3][1] = a
        test[3][2] = b
        res = np.dot(test, coeff)
        x_range = max(res[:, 0]) - min(res[:, 0])  # new image
        y_range = max(res[:, 1]) - min(res[:, 1])

        p1 = compute.Point(res[0][0], res[0][1])
        p2 = compute.Point(res[1][0], res[1][1])
        p3 = compute.Point(res[2][0], res[2][1])
        p4 = compute.Point(res[3][0], res[3][1])
        fx = (compute.Getlen(p1, p3).getlen() + compute.Getlen(p2, p4).getlen()) / 2
        fy = (compute.Getlen(p1, p2).getlen() + compute.Getlen(p3, p4).getlen()) / 2

        self.img = np.transpose(self.wrapimg, [1, 2, 0]).astype(self.wrapimg.dtype)
        pic = cv2.resize(self.img[:, :, :3], (int(fx), int(fy)))

        # angel
        ang = (res[0][1] - test[0][2]) / (res[0][0] - test[0][1])
        angel = math.atan(ang) * 180 / (math.pi)
        (h, w) = (int(x_range), int(y_range))
        center = (int(x_range / 2), int(y_range / 2))

        M = cv2.getRotationMatrix2D(center, -angel, 1.0)
        rotated = cv2.warpAffine(pic, M, (w, h))
        self.img = rotated
        cv2.imwrite(self.savepath, self.img)
        QMessageBox.information(self, "Information",
                                self.tr("校正完成!"))
        return

    def iso_action(self):
        exitAction = QAction('&ISODATA', self)
        exitAction.triggered.connect(self.isodial)
        return exitAction

    def isodial(self):
        w = QWidget()
        w.setWindowTitle('ISODATA')
        w.resize(250, 150)

        hbox1 = QHBoxLayout()  # 水平布局
        self.k0 = QLineEdit()
        self.k0.setPlaceholderText("预期聚类中心数目")
        hbox1.addWidget(self.k0)
        hbox2 = QHBoxLayout()  # 水平布局
        self.d_min = QLineEdit()
        self.d_min.setPlaceholderText("小于此距离两个聚类合并")
        hbox2.addWidget(self.d_min)
        hbox3 = QHBoxLayout()  # 水平布局
        self.sigma = QLineEdit()
        self.sigma.setPlaceholderText("样本中距离分布的标准差")
        hbox3.addWidget(self.sigma)
        hbox5 = QHBoxLayout()  # 水平布局
        self.n_min = QLineEdit()
        self.n_min.setPlaceholderText("少于此数就不作为一个独立的聚类")
        hbox5.addWidget(self.n_min)
        hbox6 = QHBoxLayout()  # 水平布局
        self.iteration = QLineEdit()
        self.iteration.setPlaceholderText("迭代次数")
        hbox6.addWidget(self.iteration)

        hbox4 = QHBoxLayout()  # 水平布局
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.ISODATA)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        hbox4.addWidget(button3)
        hbox4.addWidget(button4)

        grid = QVBoxLayout()
        grid.addLayout(hbox1)
        grid.addLayout(hbox2)
        grid.addLayout(hbox3)
        grid.addLayout(hbox5)
        grid.addLayout(hbox6)
        grid.addLayout(hbox4)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def ISODATA(self):
        img_arr = self.img
        m = img_arr.shape[0] * img_arr.shape[1]
        n = img_arr.shape[2]
        X = img_arr.reshape(m, n)

        k0 = int(self.k0.text())
        d_min = int(self.d_min.text())
        sigma = int(self.sigma.text())
        n_min = int(self.n_min.text())
        iteration = int(self.iteration.text())

        centers = CISO.randinitialize(X, k0)
        print("正在执行ISODATA算法")
        new, classes, k, result = CISO.ISODATA(X, centers, k0, d_min, sigma, n_min, iteration)
        #print("New centers:")
        #print(new)
        print("number of classes:", k)

        new_img = np.array(CISO.paint(result, img_arr))

        qimg = QImage(new_img[:, :, :3], new_img.shape[1], new_img.shape[0], new_img.shape[1] * 3,
                      QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        QMessageBox.information(self, "Information",
                                self.tr("分类完成!"))

    def Kmeans_action(self):
        exitAction = QAction('&Kmeans', self)
        exitAction.triggered.connect(self.knum)
        return exitAction

    def knum(self):
        w = QWidget()
        w.setWindowTitle('Kmeans')
        w.resize(250, 150)

        hbox1 = QHBoxLayout()  # 水平布局
        self.n = QLineEdit()
        self.n.setPlaceholderText("预期聚类中心数目")
        hbox1.addWidget(self.n)
        hbox4 = QHBoxLayout()  # 水平布局
        button3 = QPushButton('confirm')
        button3.clicked.connect(self.kmeans)
        button4 = QPushButton('cancel')
        button4.clicked.connect(w.close)
        hbox4.addWidget(button3)
        hbox4.addWidget(button4)
        grid = QVBoxLayout()
        grid.addLayout(hbox1)
        grid.addLayout(hbox4)

        w.setLayout(grid)
        self.w = w
        self.w.show()

    def kmeans(self):
        img_arr = self.img
        m = img_arr.shape[0] * img_arr.shape[1]
        n = img_arr.shape[2]
        data = img_arr.reshape(m, n)
        k = int(self.n.text())
        centers = CK.randinitialize(data, k)  # 3是预估类别数

        new, classes, result = CK.Kmeans(data, centers, k)
        #print("New centers:")
        #print(new)
        k = 0
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if result[k] == 0:
                    img_arr[i, j, :3] = [255, 0, 0]
                if result[k] == 1:
                    img_arr[i, j, :3] = [255, 250, 250]
                if result[k] == 2:
                    img_arr[i, j, :3] = [255, 255, 0]
                if result[k] == 3:
                    img_arr[i, j, :3] = [255, 192, 203]
                else:
                    pass
                k = k + 1

        new_img = Image.fromarray(img_arr)
        #new_img.show()
        new_img.save("Kmeans.jpg")

        new_img = np.array(new_img)

        qimg = QImage(new_img[:, :, :3], new_img.shape[1], new_img.shape[0], new_img.shape[1] * 3,
                      QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        QMessageBox.information(self, "Information",
                                self.tr("分类完成!"))

    def bayes_action(self):
        exitAction = QAction('&Bayes', self)
        exitAction.triggered.connect(self.bayes)
        return exitAction

    def bayes(self):
        img_arr = self.img
        m = img_arr.shape[0] * img_arr.shape[1]
        n = img_arr.shape[2]
        data = img_arr.reshape(m, n)
        centers = CK.randinitialize(data, 3)  # 预估类别数
        new, classes, result = CK.Kmeans(data, centers, 3)

        train_data = np.empty((m, n + 1))
        for i in range(m):
            for j in range(n):
                train_data[i, j] = data[i, j]
                train_data[i, j + 1] = result[i]

        test_data = data.copy()

        classOne, classTwo, classThree = NBM.spiltData(train_data)
        ave_one, devi_one = NBM.summarize(classOne)
        ave_two, devi_two = NBM.summarize(classTwo)
        ave_three, devi_three = NBM.summarize(classThree)
        prediction = NBM.predict(test_data, ave_one, devi_one, ave_two, devi_two, ave_three, devi_three)

        k = 0
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if prediction[k] == 1:
                    img_arr[i, j, :] = [255, 0, 0]
                if prediction[k] == 3:
                    img_arr[i, j, :] = [255, 250, 250]
                if prediction[k] == 2:
                    img_arr[i, j, :] = [255, 255, 0]
                else:
                    pass
                k = k + 1

        new_img = Image.fromarray(img_arr)
        new_img.save("Bayes.jpg")

        new_img = np.array(new_img)

        qimg = QImage(new_img[:, :, :3], new_img.shape[1], new_img.shape[0], new_img.shape[1] * 3,
                      QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        QMessageBox.information(self, "Information",
                                self.tr("分类完成!"))



if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    ex = RSImgPro()
    sys.exit(app.exec_())


