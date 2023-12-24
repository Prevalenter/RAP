import sys
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,\
    QRadioButton, QGroupBox, QGridLayout, QPlainTextEdit, QSpinBox, QComboBox, QDoubleSpinBox

import sys
sys.path.append('..')
from connect.client import Client

# pre_action_dict = {
#     'back zero': [0.4, 0.0, 0.8, 3.14159, 0, 3.14159],
#     # 'to hole': [0.6363, 0.15408, 0.18055, 3.14159, 0, 3.14159],
#     'to hole': [0.6363, 0.15408, 0.190, 3.14159, 0, 3.14159]
# }

class StateWidget(QWidget):
    def __init__(self, up_ctrl=None):
        super().__init__()

        self.up_ctrl = up_ctrl
        self.ui_init()

    def ui_init(self):
        state_box = QGroupBox("Robot state")
        layout_state = QGridLayout()
        self.angle_le_list = []

        layout_state.addWidget(QLabel("Joint"), 0, 0)
        layout_state.addWidget(QLabel("XYZ-Rot"), 1, 0)
        layout_state.addWidget(QLabel("FT in Sensor"), 2, 0)
        layout_state.addWidget(QLabel("FT in Word"), 3, 0)

        self.state_le_list = {"joints": [],
                              "xyz": [],
                              "ft": [],
                              "ft_world": []}
        for idx, type in enumerate(self.state_le_list.keys()):
            for j in range(6):
                self.state_le_list[type].append( QLineEdit("") )
                layout_state.addWidget( self.state_le_list[type][-1], idx, j+1)

        state_box.setLayout(layout_state)
        # state_box.setMaximumHeight(240)

        container = QVBoxLayout()
        container.addWidget(state_box)
        container.addStretch(2)
        self.setLayout(container)

    def update(self, joints=None, xyz=None, ft=None, ft_world=None):
        if joints is not None:
            for i in range(6):
                # self.state_le_list['joints'][i].set =
                self.state_le_list['joints'][i].setText("%.4f" % (joints[i]))

        if xyz is not None:
            for i in range(6):
                self.state_le_list['xyz'][i].setText("%.4f" % (xyz[i]))

        if ft is not None:
            for i in range(6):
                self.state_le_list['ft'][i].setText("%.4f" % (ft[i]))

        if ft_world is not None:
            for i in range(3):
                self.state_le_list['ft_world'][i].setText("%.4f" % (ft_world[i]))

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.state_widget = StateWidget(up_ctrl=self)
        self.state_widget.setParent(self)

        self.resize(900, 150)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()