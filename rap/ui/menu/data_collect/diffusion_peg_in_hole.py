
import numpy as np
import time
import sys
sys.path.append('../../../')
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pickle

from ui.base.vedio import VedioWidget
from ui.base.state import StateWidget

class DiffusionPegInHoleWidget(QDialog):
    def __init__(self, single_cam_list, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl

        self.step_index = 0
        # self.dt = 0.05

        self.single_cam_list = single_cam_list

        self.ui_init()

    def ui_init(self):
        container = QVBoxLayout()

        self.state_widget = StateWidget()
        container.addWidget(self.state_widget)

        self.cam_list = []
        for idx in range(len(self.single_cam_list)):
            cam = VedioWidget(single_cam=self.single_cam_list[idx])
            cam.setParent(self)

            container.addWidget(cam)
        container.addStretch(2)
        self.setLayout(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)

    def update(self):
        print('diffusion peg in hole update')
        if self.up_ctrl is not None:
            self.state_widget.update(self.up_ctrl.angle_cur,
                                     self.up_ctrl.xyz_cur,
                                     self.up_ctrl.ft.ft_sensor,
                                     self.up_ctrl.ft.ft_contact)

    def showEvent(self, a0):
        print('diffusion show event')
        self.timer.start(200)

    def closeEvent(self, a0):
        print('diffusion close event')
        self.timer.stop()


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.state_widget = DiffusionPegInHoleWidget(up_ctrl=self)
        self.state_widget.setParent(self)

        self.resize(1000, 800)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()


