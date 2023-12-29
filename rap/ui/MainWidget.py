import os
import sys
sys.path.append("..")

from ui.ui3d.axis import Axis
from ui.ui3d.forcesensor import ForceSensor

import matplotlib
matplotlib.use('Qt5Agg')
from plot3d import Ui3dWindow
from plot2d import Plot2dWindow, PlotMainWindow
from state import StateWidget
from connet_widget import QApplication, ConnectWidget
from ui.calibrate_widget import CalibrateWidget
# from control_widget import ControlWidget
from ui.setting import ParaSetWidget
from utils import force_adapter

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication, QTabWidget, QMainWindow


class MainWidget(QWidget):
# class MainWindow(QWi):
    def __init__(self, ui_type='3d'):
        super().__init__()
        self.xyz_tgt = [0]*6
        self.angle_tgt = [0]*6
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        para_file_root = "../data/para/"
        self.para_magager = {}
        for fn in os.listdir(para_file_root):
            print( para_file_root, fn )
            self.para_magager[fn.split('.')[0]] = ParaSetWidget(path=para_file_root+fn)
        # self.para_set =


        self.pre_action_dict = {
            'back zero': [0.4, 0.0, 0.8, 3.14159, 0, 3.14159],
            'compliance zero': [0.4, 0.2, 0.6, 3.14159, 0.0, 3.14159],
            'contact zero': [0.6363, 0.130, 0.20, 3.14159, 0.0, 3.14159],
            # 'test': [0.4000158067460178, 0.0, 0.8000742610857445, 3.14159, 0.0, 3.14159],
            # 'to hole': [0.6363, 0.15408, 0.18055, 3.14159, 0, 3.14159],
            'to hole': [0.6363, 0.15408, 0.20, 3.14159, 0, 3.14159]
        }

        self.axis = Axis(scale=0.1)
        # self.axis.set_axis(angles=[0.3, 0, 0])
        self.axis_ori = Axis()
        self.ft = ForceSensor()

        self.force_flag_dict = {
            'Drag': False,
            'Compliance': False,
            'Position Force': False,
            'None': True
        }
        # self.compliance = False
        # self.drag_force_flag = False

        # self.cartesian_work_space= [[0.25, 0.65], # x
        #                             [0.00, 0.50], # y
        #                             [0.20, 1.00], # z
        #                             [-0.60, 0.60], # rot x
        #                             [-0.60, 0.60], # rot y
        #                             [-3.14, 3.14] # rot z
        #                            ]


        self.control_tab = QTabWidget()
        self.connect_widget = ConnectWidget(self)
        self.calibrate_widget = CalibrateWidget(self)
        self.control_tab.addTab(self.connect_widget, "Connect")
        self.control_tab.addTab(self.calibrate_widget, "calibrate")
        # self.control_widget = ControlWidget(up_ctrl=self)
        # self.control_widget.setParent(self)

        # self.connect_widget = self.control_widget.connect

        self.state_widget = StateWidget(up_ctrl=self)


        if ui_type=='3d':
            self.plot = Ui3dWindow(up_ctrl=self)
            self.plot.setParent(self)
            self.plot.setMinimumHeight(800)
        else:
            self.plot = Plot2dWindow(up_ctrl=self)

        self.drag_fa = force_adapter.DragForceAdapter(self)
        self.compliance_fa = force_adapter.ComplianceForceAdapter(self)
        self.position_force_fa = force_adapter.PositionForceAdapter(self)

        # self.compliance_fa.start()

        layout = QHBoxLayout()

        layout_mid = QVBoxLayout()
        layout_mid.addWidget(self.plot)
        layout_mid.addWidget(self.state_widget)
        self.control_tab.setMaximumWidth(300)
        layout.addWidget(self.control_tab)
        layout.addLayout(layout_mid)

        self.setLayout(layout)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            print('update tgt!')
            self.connect_widget.update_tgt()
        if sent_tgt:
            self.connect_widget.apply_xyz()
        self.plot.update()
        self.state_widget.update(self.angle_cur, self.xyz_cur, self.ft.ft_sensor, self.ft.ft_contact)

        # print('UI update down')
        # self.plot2d.update()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MainWidget()
    # w = ConnectWidget()
    w.show()
    app.exec_()