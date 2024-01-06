
import numpy as np
import time
import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pickle

class DiffusionPegInHoleWidget(QDialog):
    def __init__(self, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl

        self.step_index = 0
        # self.dt = 0.05

        self.para = {
            'ylim': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            'data': [],
            'dt': 0.050,
            'start': False,
        }
        self.ui_init()

    def test_data(self):
        t = self.para['dt'] * self.step_index
        return [np.sin(t+i) for i in range(6)]

    def ui_init(self):
        state_box = QGroupBox("Dataset sample")
        layout_state = QGridLayout()
        self.angle_le_list = []

        layout_state.addWidget(QLabel("Realsense"), 0, 0)
        layout_state.addWidget(QLabel("Ylim Low"), 1, 0)

        self.state_le_list = {"Ylim Upper": [],
                              "Ylim Lower": []}
        # for idx, type in enumerate(self.state_le_list.keys()):
        #     for j in range(6):
        #         self.state_le_list[type].append( QDoubleSpinBox() )
        #         self.state_le_list[type][-1].setMinimum(-1000)
        #         self.state_le_list[type][-1].setMaximum(1000)
        #         # QDoubleSpinBox().setValue()
        #         if type=="Ylim Upper":
        #             self.state_le_list[type][j].setValue(self.para['ylim'][j][1])
        #         else:
        #             self.state_le_list[type][j].setValue(self.para['ylim'][j][0])
        #         # QSpinBox
        #         layout_state.addWidget( self.state_le_list[type][-1], idx, j+1)

        self.btn_set = QPushButton('Set')
        self.btn_start = QPushButton('Start')
        # self.btn_stop = QPushButton('Stop')
        # self.btn_load = QPushButton('Load')
        # self.btn_save = QPushButton('Save')
        # self.btn_replay = QPushButton('Replay')
        # self.btn_clear = QPushButton('Clear')
        # self.check_plot = QCheckBox("Plot")
        # layout_state.addWidget(self.check_plot, 3, 0)
        # layout_state.addWidget(self.btn_set, 3, 1)
        # layout_state.addWidget(self.btn_load, 3, 2)
        # layout_state.addWidget(self.btn_save, 3, 3)
        # layout_state.addWidget(self.btn_replay, 3, 4)
        # layout_state.addWidget(self.btn_clear, 3, 5)
        # layout_state.addWidget(self.btn_start, 3, 6)
        # layout_state.addWidget(self.btn_stop, 3, 7)

        state_box.setLayout(layout_state)
        # state_box.setMaximumHeight(240)

        container = QVBoxLayout()
        container.addWidget(state_box)
        # container.addWidget(self.slider)
        container.addStretch(2)
        self.setLayout(container)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.state_widget = DiffusionPegInHoleWidget(up_ctrl=self)
        self.state_widget.setParent(self)

        self.resize(800, 300)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()


