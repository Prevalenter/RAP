import sys
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,\
    QRadioButton, QGroupBox, QGridLayout, QPlainTextEdit, QSpinBox, QComboBox, QDoubleSpinBox, QDialog

import copy

import sys
sys.path.append('..')
from utils.json_para import save_para, load_para


class ParaSetWidget(QDialog):
    def __init__(self, up_ctrl=None):
        super().__init__()

        self.path = '../data/para.json'

        self.up_ctrl = up_ctrl
        self.para = load_para(self.path)
        self.para_widget = copy.deepcopy(self.para)
        print('load the para is: ')
        print(self.para)

        self.ui_init()

    def ui_init(self):
        container = QVBoxLayout()
        print(list(self.para.keys()))
        for group_name in self.para.keys():
            print(group_name)
            state_box = QGroupBox(group_name)
            layout_state = QGridLayout()
            for para_idx, para_name in enumerate(self.para[group_name]):
                para_type = type(self.para[group_name][para_name])
                layout_state.addWidget(QLabel(para_name), para_idx, 0)
                if para_type is list:
                    print(f"{para_name} is list!")
                    para_widget = []
                    for i, v in enumerate(self.para[group_name][para_name]):
                        para_widget.append(QDoubleSpinBox())
                        para_widget[-1].setMinimum(-1e5)
                        para_widget[-1].setMaximum(1e5)
                        para_widget[-1].setSingleStep(0.01)
                        para_widget[-1].setValue(v)
                        layout_state.addWidget(para_widget[-1], para_idx, 1+i)
                self.para_widget[group_name][para_name] = para_widget
                self.para_widget[group_name]['Set'] = QPushButton("Set")

                self.para_widget[group_name]['Set'].clicked.connect(self.on_set_para)

                layout_state.addWidget(self.para_widget[group_name]['Set'], para_idx+1, 1)

                state_box.setLayout(layout_state)
            state_box.setMaximumHeight(240)


            container.addWidget(state_box)
            container.addStretch(2)
        self.setLayout(container)


    def on_set_para(self):
        print("set para")
        for group_name in self.para.keys():
            print(group_name)
            for para_idx, para_name in enumerate(self.para[group_name]):
                para_type = type(self.para[group_name][para_name])
                print(para_idx, para_name, para_type)

                if para_type is list:
                    value_list = []
                    for weidget in self.para_widget[group_name][para_name]:
                        # print( dir(weidget) )
                        value_list.append( weidget.value() )
                    self.para[group_name][para_name] = value_list
                    print(value_list)


        print('-'*50)
        print(self.para)

        save_para(self.path, self.para)



class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.state_widget = ParaSetWidget(up_ctrl=self)
        self.state_widget.setParent(self)

        self.resize(800, 400)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    # w = ConnectWidget()
    w.show()
    app.exec_()