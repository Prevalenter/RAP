import sys
sys.path.append("..")

import matplotlib
matplotlib.use('Qt5Agg')
from plot3d import Ui3dWindow
from plot2d import Plot2dWindow, PlotMainWindow
from connet_widget import QApplication, ConnectWidget
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

class MainWindow(QWidget):
    def __init__(self, ui_type='3d'):
        super().__init__()
        self.xyz_tgt = [0]*6
        self.angle_tgt = [0]*6
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.connect_widget = ConnectWidget(up_ctrl=self)
        self.connect_widget.setParent(self)
        if ui_type=='3d':
            self.plot = Ui3dWindow(up_ctrl=self)
            self.plot.setParent(self)
        else:
            self.plot = Plot2dWindow(up_ctrl=self)

        layout = QHBoxLayout()

        layout.addWidget(self.connect_widget)
        layout.addWidget(self.plot)

        self.setLayout(layout)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.connect_widget.update_tgt()
        if sent_tgt:
            self.connect_widget.apply_xyz()
        self.plot.update()
        # self.plot2d.update()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MainWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()