import sys
import cv2
import numpy as np
from osgeo import gdalnumeric
# 这里我们提供必要的引用。基本控件位于pyqt5.qtwidgets模块中。
from PyQt5.QtCore import Qt,pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication,QWidget,QToolTip,QPushButton,QMessageBox,QDesktopWidget, QLabel,\
    QHBoxLayout, QVBoxLayout,QGridLayout, QLineEdit, QTextEdit,QMainWindow, QAction, qApp, QLCDNumber, \
    QSlider, QInputDialog,QFrame, QColorDialog,QFileDialog
from PyQt5.QtGui import QIcon,QFont,QColor,QPixmap,QImage
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtCore import QCoreApplication
num = 0
ban = []
wrapp = []

class Window(QMainWindow):
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
        self.label=QLabel('RS image2')
        self.label.resize(450,450)
        grid.addWidget(self.label,0,0)
        w.setLayout(grid)
        menubar = self.menuBar()

        # 添加菜单
        self.subwidget = w
        self.setCentralWidget(self.subwidget)
        fileMenu = menubar.addMenu('&File')

        # 添加事件
        openfile_action = self.open_action()  # file
        fileMenu.addAction(openfile_action)

        self.setGeometry(300, 300,300,300)
        self.setWindowTitle('Window')
        self.setWindowIcon(QIcon('1.jpg'))
        #self.show()

    def mousePressEvent(self, QMouseEvent):
        print(QMouseEvent.pos())
        global wrapp
        wrapp.append(QMouseEvent.pos())
        self.wrapp = wrapp

    def openfileDialog(self):                                                                                           #open image
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        #self.cho.show()
        if fname[0]:
            img = gdalnumeric.LoadFile(fname[0])                                     #6*400*640
            #img = np.array(img)
            if len(img.shape)==3:
                global num
                num = img.shape[0]
                self.img = np.transpose(img, [1, 2, 0]).astype(img.dtype)                #400*640*6
                self.choosebands()
            else:
                self.img = img
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

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event):  # this is the rewrite of the function
        if self.ctrlPressed:  # if the ctrl key is pressed: then deal with the defined process
            delta = event.angleDelta()
            oriention = delta.y() / 8
            self.pixmap.zoomsize = 0
            if oriention > 0:
                self.pixmap.zoomsize += 1
            else:
                self.pixmap.zoomsize -= 1
            self.pixmap.zoomIn(self.pixmap.zoomsize)
            print(self.pixmap.zoomsize)
        else:  # if the ctrl key isn't pressed then submiting                   the event to it's super class
            return super().wheelEvent(event)


if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())