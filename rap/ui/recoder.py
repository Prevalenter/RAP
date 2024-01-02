
import numpy as np
import time
import sys
import numpy as np
from PyQt5.QtWidgets import *

from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

import pickle

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = self.figure.add_subplot(611)
        self.axes = self.figure.subplots(6, 1)
        super(MplCanvas, self).__init__(self.figure)


class ForcePlotdWindow(QMainWindow):
    def __init__(self, up_ctrl=None, sapce_range=[[0.270, 0.650], [0.030, 0.500], [0.300, 0.600]], *args, **kwargs):
        super(ForcePlotdWindow, self).__init__(*args, **kwargs)

        self.up_ctrl = up_ctrl

        self.x = np.array(20*10)*0.05
        self.data = []

        self.canvas = MplCanvas(self, width=9, height=7)
        self.axes = self.canvas.axes
        self.setCentralWidget(self.canvas)

        self.show()
        self.sapce_range = sapce_range



    def update(self, para):

        data = para['data']
        ylim = para['ylim']

        for i in range(6):
            self.axes[i].clear()
            data_len = len(data)
            slider_idx = para['slider_value']
            if slider_idx<200:
                data_draw = np.array(data)
            else:
                print(data_len, slider_idx)
                data_draw = np.array(data[slider_idx-200:slider_idx])
            print(data_len, data_draw.shape)
            if data_len==0: continue
            self.axes[i].plot(np.arange(data_draw.shape[0])*para['dt'], data_draw[:, i], c='k', lw=1)
            self.axes[i].set_xlim(0, 10)
            # self.axes[i].set_ylim(-10, 10)

            ylim_i = ylim[i]
            if ylim_i[0]<ylim_i[1]:
                self.axes[i].set_ylim(ylim_i[0], ylim_i[1])

        label_list = ['x', 'y', 'z']
        for i in range(3):
            self.axes[i].set_ylabel(f'F_{label_list[i]} (N)')
            self.axes[i+3].set_ylabel(f'T_{label_list[i]} (Nm)')
        self.axes[5].set_xlabel('Time (s)')
        self.canvas.draw()

        self.canvas.figure.subplots_adjust(bottom=0.1,top=0.95, right=0.92, hspace=0.1)


class FTRecoderWidget(QDialog):
    def __init__(self, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl


        self.step_index = 0
        # self.dt = 0.05

        self.para = {
            'ylim': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            'data': [],
            'dt': 0.050,
            'start': True,
            'slider_value': 0
        }
        self.ui_init()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        # self.timer.start(int(1000*self.para['dt']))
        self.timer.start(50)

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update)
        self.plot_timer.start(500)


    def update(self):
        t = time.time()
        self.plot2d_panel.update(self.para)
        len_data = len(self.para['data'])
        if len_data>200:
            self.slider.setMinimum(200)
        self.slider.setMaximum(len_data)
        if self.para['start']: self.slider.setValue(len_data)
        print('step using: ', time.time()-t)

    def step(self):
        t = time.time()
        if self.para['start']:
            data_i = self.test_data()
            self.para['data'].append(data_i)
        self.step_index += 1
        print(time.time()-t)



    def test_data(self):
        t = self.para['dt'] * self.step_index
        return [np.sin(t+i) for i in range(6)]

    def ui_init(self):
        state_box = QGroupBox("Reocder Setting")
        layout_state = QGridLayout()
        self.angle_le_list = []

        layout_state.addWidget(QLabel("Ylim Up"), 0, 0)
        layout_state.addWidget(QLabel("Ylim Low"), 1, 0)

        self.state_le_list = {"Ylim Upper": [],
                              "Ylim Lower": []}
        for idx, type in enumerate(self.state_le_list.keys()):
            for j in range(6):
                self.state_le_list[type].append( QDoubleSpinBox() )
                self.state_le_list[type][-1].setMinimum(-1000)
                self.state_le_list[type][-1].setMaximum(1000)
                # QDoubleSpinBox().setValue()
                if type=="Ylim Upper":
                    self.state_le_list[type][j].setValue(self.para['ylim'][j][1])
                else:
                    self.state_le_list[type][j].setValue(self.para['ylim'][j][0])
                # QSpinBox
                layout_state.addWidget( self.state_le_list[type][-1], idx, j+1)

        self.btn_set = QPushButton('Set')
        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_load = QPushButton('Load')
        self.btn_save = QPushButton('Save')
        self.btn_replay = QPushButton('Replay')
        self.btn_clear = QPushButton('Clear')
        layout_state.addWidget(self.btn_set, 3, 1)
        layout_state.addWidget(self.btn_load, 3, 2)
        layout_state.addWidget(self.btn_save, 3, 3)
        layout_state.addWidget(self.btn_replay, 3, 4)
        layout_state.addWidget(self.btn_clear, 3, 5)
        layout_state.addWidget(self.btn_start, 3, 6)
        layout_state.addWidget(self.btn_stop, 3, 7)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_set.clicked.connect(self.on_set)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_clear.clicked.connect(self.on_clear)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)

        state_box.setLayout(layout_state)
        # state_box.setMaximumHeight(240)

        self.plot2d_panel = ForcePlotdWindow(self)

        container = QVBoxLayout()
        container.addWidget(state_box)
        container.addWidget(self.slider)
        container.addWidget(self.plot2d_panel)
        container.addStretch(2)
        self.setLayout(container)

    def on_slider(self):
        self.para['slider_value'] = self.slider.value()
        print(self.para['slider_value'])
        self.plot2d_panel.update(self.para)


    def on_load(self):
        print('on load')
        # fileName_choose = 'test.pkl'
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "File save",
                                                                "/home/lx/Documents/lx/work/hust/RAP/rap/data",
                                                                options=QFileDialog.DontUseNativeDialog)
        if fileName_choose!="":
            with open(fileName_choose, "rb") as f:
                self.para = pickle.load(f)

            self.slider.setMaximum(len(self.para['data']))
            self.slider.setValue(len(self.para['data']))

            self.on_set()
            self.para['start'] = False
            self.plot2d_panel.update(self.para)

    def on_save(self):
        print('on save')
        # fileName_choose = 'test.pkl'
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,
                                                                "File save",
                                                                "/home/lx/Documents/lx/work/hust/RAP/rap/data",
                                                                options=QFileDialog.DontUseNativeDialog)
        if fileName_choose!="":
            print('save file in: ', fileName_choose)

            # np.save(fileName_choose, np.zeros(6))
            with open(fileName_choose, "wb") as f:
                pickle.dump(self.para, f)


    def on_set(self):
        for idx, type in enumerate(self.state_le_list.keys()):
            for j in range(6):
                if type=="Ylim Upper":
                    # self.state_le_list[type][j].setValue(self.para['ylim'][j][1])
                    self.para['ylim'][j][1] = self.state_le_list[type][j].value()
                else:
                    # self.state_le_list[type][j].setValue(self.para['ylim'][j][0]
                    self.para['ylim'][j][0] = self.state_le_list[type][j].value()


    def on_start(self):
        # self.timer.start(int(1000*self.para['dt']))
        self.para['start'] = True

    def on_stop(self):
        print('on stop')
        # self.timer.stop()
        self.para['start'] = False

    def on_clear(self):
        self.para['data'] = []
        self.plot2d_panel.update(self.para)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.state_widget = RecoderWidget(up_ctrl=self)
        self.state_widget.setParent(self)

        self.resize(1000, 900)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()


