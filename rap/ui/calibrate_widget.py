import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,\
    QRadioButton, QGroupBox, QGridLayout, QPlainTextEdit, QSpinBox, QComboBox, QDoubleSpinBox

# from PyQt5 import QtTest

from PyQt5.QtCore import QTimer

import sys
sys.path.append('..')

class CalibrateWidget(QWidget):
    def __init__(self, up_ctrl=None):
        super().__init__()

        self.client = None
        self.up_ctrl = up_ctrl

        self.data_xyz_init = False
        self.data_agnle_init = False

        self.ft_traj = []

        self.ui_init()

    def ui_init(self):
        self.resize(240, 1080)
        self.setMaximumWidth(300)

        self.setWindowTitle("layout 1")

        layout_ip = QGridLayout()

        ip_box = QGroupBox("Force-Torque")
        # l1 = QLabel("IP: ")
        # l2 = QLabel("Port:")
        #
        # layout_ip.addWidget(l1, 0, 0)
        # layout_ip.addWidget(l2, 1, 0)
        #
        # le1 = QLineEdit("localhost")
        # le2 = QLineEdit("12340")
        #
        # layout_ip.addWidget(le1, 0, 1)
        # layout_ip.addWidget(le2, 1, 1)

        self.btn_get_data = QPushButton('Get Ident Data')
        self.btn_get_data.clicked.connect(self.on_get_data)
        # self.btn_disconnect = QPushButton('Disconnect')
        layout_ip.addWidget(self.btn_get_data, 0, 0)
        # layout_ip.addWidget(self.btn_disconnect, 3, 1)

        ip_box.setLayout(layout_ip)
        ip_box.setMaximumHeight(150)

        container = QVBoxLayout()
        container.addWidget(ip_box)

        container.addStretch(2)
        self.setLayout(container)


    def on_get_data(self):

        self.timer_recoder = QTimer(self)
        self.timer_recoder.timeout.connect(self.step_recoder)
        self.timer_recoder.start(100)

        self.ft_traj = []

        timer = QTimer(self)
        timer.singleShot(0, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, 0.5, 0, 0]))
        timer.stop()

        timer = QTimer(self)
        timer.singleShot(4000, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, 0.5, 0.5, 0]))
        timer.stop()

        timer = QTimer(self)
        timer.singleShot(8000, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, 0.5, 0.5, 0.5]))
        timer.stop()

        timer = QTimer(self)
        timer.singleShot(12000, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, -0.5, 0, 0]))
        timer.stop()

        timer = QTimer(self)
        timer.singleShot(16000, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, -0.5, 0.0, -0.5]))
        timer.stop()

        timer = QTimer(self)
        timer.singleShot(20000, lambda : self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, -0.5, -0.5, -0.5]))
        timer.stop()

    def step_recoder(self):
        print("step pick", len(self.ft_traj))
        if self.up_ctrl is not None and self.up_ctrl.connect_widget.client is not None:
            self.ft_traj.append([self.up_ctrl.ft_cur, self.up_ctrl.xyz_cur])
        else:
            self.ft_traj.append([[0]*6, [0]*6])

        if len(self.ft_traj)>240:
            self.timer_recoder.stop()
            print('save!')
            np.save('../data/calibrate/ft.npy', np.array(self.ft_traj))


    def save(self):
        pass

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_tgt = [0]*6
        self.angle_tgt = [0]*6
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.connect_widget = CalibrateWidget(up_ctrl=self)
        self.connect_widget.setParent(self)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.connect_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()