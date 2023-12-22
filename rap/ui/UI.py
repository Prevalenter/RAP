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

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication, QTabWidget

class MainWindow(QWidget):
    def __init__(self, ui_type='3d'):
        super().__init__()
        self.xyz_tgt = [0]*6
        self.angle_tgt = [0]*6
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.axis = Axis(scale=0.1)
        # self.axis.set_axis(angles=[0.3, 0, 0])
        self.axis_ori = Axis()
        self.ft = ForceSensor()


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
            self.plot.setMinimumHeight(850)
        else:
            self.plot = Plot2dWindow(up_ctrl=self)

        layout = QHBoxLayout()

        layout_mid = QVBoxLayout()
        layout_mid.addWidget(self.plot)
        layout_mid.addWidget(self.state_widget)

        layout.addWidget(self.control_tab)
        layout.addLayout(layout_mid)

        self.setLayout(layout)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.connect_widget.update_tgt()
        if sent_tgt:
            self.connect_widget.apply_xyz()
        self.plot.update()
        self.state_widget.update(self.angle_cur, self.xyz_cur, self.ft.ft_sensor)
        # self.plot2d.update()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MainWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()