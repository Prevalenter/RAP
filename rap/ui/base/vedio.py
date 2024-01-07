import numpy as np

import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap
import sys


import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

import sys
sys.path.append('../../')

from utils.realsense import SingleReansense, get_dev_id_list

# dev_id_list = ['147122075207',
#                '147322070524']



class CamThread(QThread):
    # def __init__(self, dev_id):
    changePixmap = pyqtSignal(QImage)

    # def get_cam(self, dev_id):
    #     self.single_cam = SingleReansense(dev_id)
    #     self.fps = 0

    def get_cam(self, single_cam):
        self.single_cam = single_cam
        self.fps = 0

    def run(self):
        fps_list = []
        while True:
            t = time.time()
            img = self.single_cam.get_frame()

            h, w, c = img.shape
            convertToQtFormat = QImage(img.data, w, h, QImage.Format_RGB888)
            convertToQtFormat = convertToQtFormat.scaled(640, 480//2, Qt.KeepAspectRatio)
            self.changePixmap.emit(convertToQtFormat)

            fps_list.append( 1/(time.time()-t) )

            if len(fps_list)==10:
                self.fps = np.array(fps_list).mean()
                fps_list = []

class VedioWidget(QDialog):
    def __init__(self, single_cam, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl
        self.single_cam = single_cam

        self.step_index = 0
        self.ui_init()

    def ui_init(self):

        self.setGeometry(0, 0, 640, 480//2)

        container = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setFixedWidth(640)
        self.label.setFixedHeight(480//2)

        self.label_txt = QLabel("FPS: 0.00")

        # self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.green)
        # self.setPalette(p)

        container.addWidget(self.label)
        container.addWidget(self.label_txt)
        self.setLayout(container)

        self.th = CamThread(self)
        self.th.get_cam(self.single_cam)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()

        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(500)  # every 10,000 milliseconds

    def update(self):
        # print('update')
        self.label_txt.setText("FPS: %.2f"%(self.th.fps))

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

class MyWindow123(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0] * 6
        self.angle_cur = [0] * 6
        self.ft_cur = [0] * 6

        # self.resize(640, 480)
        single_cam = SingleReansense(dev_id='147122075207')

        self.vedio_widget = VedioWidget(single_cam, up_ctrl=self)
        self.vedio_widget.setParent(self)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.blue)
        self.setPalette(p)


        # self.resize(640, 480)
        # print(self.size())

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.vedio_widget.update_tgt()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow123()
    # w = ConnectWidget()
    w.show()
    app.exec_()

