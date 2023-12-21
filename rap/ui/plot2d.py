import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from PyQt5.QtWidgets import QApplication, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)


class Plot2dWindow(QtWidgets.QMainWindow):

    def __init__(self, up_ctrl=None, sapce_range=[[0.270, 0.650], [0.030, 0.500], [0.300, 0.600]], *args, **kwargs):
        super(Plot2dWindow, self).__init__(*args, **kwargs)

        # sc = MplCanvas(self, width=5, height=4)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.canvas = MplCanvas(self, width=5, height=4)
        self.axes = self.canvas.axes

        self.setCentralWidget(self.canvas)

        self.show()

        self.up_ctrl = up_ctrl

        self.sapce_range = sapce_range

        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.update()

    def set_xyz_tgt(self, xy):
        self.up_ctrl.xyz_tgt[:2] = xy
        # self.draw()
        self.up_ctrl.update(tgt=True, sent_tgt=True)

    def update(self):
        self.axes.clear()
        self.axes.scatter(*self.up_ctrl.xyz_tgt[:2], c='g', label='tgt')
        self.axes.scatter(*self.up_ctrl.xyz_cur[:2], marker="+", c='b', label='cur')

        self.axes.set_xlim(self.sapce_range[0][0], self.sapce_range[0][1])
        self.axes.set_ylim(self.sapce_range[1][0], self.sapce_range[1][1])
        self.axes.set_xlabel('x', fontsize=16)
        self.axes.set_ylabel('y', fontsize=16)

        self.axes.legend()
        self.canvas.draw()

        self.canvas.figure.subplots_adjust(bottom=0.15, top=0.92, right=0.92)

    def on_click(self, event):
        # print(dir(event))
        if event.button is MouseButton.LEFT:
            if event.xdata is not None:
                self.set_xyz_tgt([event.xdata, event.ydata])

class PlotMainWindow(QWidget):
    def __init__(self, up_ctrl=None):
        super().__init__()
        self.xyz_tgt = [0.5, 0.3, 0, 0, 0, 0]
        self.xyz_cur = [0.45, 0.1, 0, 0, 0, 0]
        self.plot2d_panel = Plot2dWindow(self)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        self.plot2d_panel.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = PlotMainWindow()
    # w = ConnectWidget()
    # w.show()
    app.exec_()