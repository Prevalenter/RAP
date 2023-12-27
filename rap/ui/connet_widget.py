import sys
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,\
    QRadioButton, QGroupBox, QGridLayout, QPlainTextEdit, QSpinBox, QComboBox, QDoubleSpinBox, QCheckBox

import sys
sys.path.append('..')
from connect.client import Client



class ConnectWidget(QWidget):
    def __init__(self, up_ctrl=None):
        super().__init__()

        self.client = None
        self.up_ctrl = up_ctrl

        self.data_xyz_init = False
        self.data_agnle_init = False

        self.ft_traj = []


        self.ui_init()

    def ui_init(self):
        self.resize(240, 900)
        self.setMaximumWidth(300)

        self.setWindowTitle("layout 1")

        layout_ip = QGridLayout()

        ip_box = QGroupBox("IP Connect")
        l1 = QLabel("IP: ")
        l2 = QLabel("Port:")

        layout_ip.addWidget(l1, 0, 0)
        layout_ip.addWidget(l2, 1, 0)

        le1 = QLineEdit("localhost")
        le2 = QLineEdit("12340")

        layout_ip.addWidget(le1, 0, 1)
        layout_ip.addWidget(le2, 1, 1)

        self.btn_connect = QPushButton('Connect')
        self.btn_disconnect = QPushButton('Disconnect')
        layout_ip.addWidget(self.btn_connect, 2, 1)
        layout_ip.addWidget(self.btn_disconnect, 3, 1)

        self.btn_connect.clicked.connect(self.on_connect)
        # self.btn_connect.click()

        ip_box.setLayout(layout_ip)
        ip_box.setMaximumHeight(150)

        preset_box = QGroupBox("Preset action")
        preset_layout = QGridLayout()

        self.btn_radio_follow = QRadioButton('Follow')
        self.btn_radio_ft_sensor = QRadioButton('FT Sensor')
        # btn3 = QRadioButton('curl hair')

        self.action_combo = QComboBox()
        self.action_combo.addItems(list(self.up_ctrl.pre_action_dict))
        self.btn_preaction = QPushButton('Apply')
        self.btn_preaction.clicked.connect(self.on_apply_preset)

        self.btn_radio_follow.setChecked(True)

        preset_layout.addWidget(self.btn_radio_follow, 0, 0)
        preset_layout.addWidget(self.btn_radio_ft_sensor, 0, 1)
        preset_layout.addWidget(self.action_combo, 1, 0)
        preset_layout.addWidget(self.btn_preaction, 1, 1)
        self.btn_radio_follow.toggled.connect(lambda : self.on_mode_radio_btn(self.btn_radio_follow))
        self.btn_radio_ft_sensor.toggled.connect(lambda: self.on_mode_radio_btn(self.btn_radio_ft_sensor))
        preset_box.setLayout(preset_layout)
        preset_box.setMaximumHeight(120)


        control_box = QGroupBox("Control set")
        control_layout = QGridLayout()
        # self.check_drag_force = QCheckBox('Drag')
        # self.check_compliance_force = QCheckBox('Compliance')

        # Drag Compliance
        self.btn_radio_drag_force = QRadioButton('Drag')
        self.btn_radio_compliance_force = QRadioButton('Compliance')
        self.btn_radio_none = QRadioButton('None')


        self.btn_radio_drag_force.toggled.connect(lambda: self.on_force_radio_btn(self.btn_radio_drag_force))
        self.btn_radio_compliance_force.toggled.connect(lambda: self.on_force_radio_btn(self.btn_radio_compliance_force))
        self.btn_radio_none.toggled.connect(lambda: self.on_force_radio_btn(self.btn_radio_none))

        self.btn_radio_none.setChecked(True)

        control_layout.addWidget(self.btn_radio_drag_force, 0, 0)
        control_layout.addWidget(self.btn_radio_compliance_force, 0, 1)
        control_layout.addWidget(self.btn_radio_none, 0, 2)
        control_box.setLayout(control_layout)


        # state_box = QGroupBox("Robot state")
        # layout_state = QGridLayout()
        #
        # self.angle_le_list = []
        #
        # for i in range(2):
        #     for j in range(3):
        #         layout_state.addWidget(QLabel(f"Joint {2*i+j+1}"), i*2, j)
        #         self.angle_le_list.append(QLineEdit(""))
        #         layout_state.addWidget(self.angle_le_list[-1], i*2+1, j)
        #
        # self.xyz_le_list = []
        # for idx, l in enumerate("XYZABC"):
        #
        #     layout_state.addWidget(QLabel(l), 4+(idx//3)*2, idx%3)
        #     self.xyz_le_list.append(QLineEdit(""))
        #     layout_state.addWidget(self.xyz_le_list[-1], 4+(idx//3)*2+1, idx%3)
        #
        # state_box.setLayout(layout_state)
        # state_box.setMaximumHeight(240)

        angle_set_box = QGroupBox("Angle Set")
        layout_set_angle = QGridLayout()
        self.angle_spin_list = []
        for i in range(6):
            layout_set_angle.addWidget(QLabel(f"Joint {i}"), i, 0)
            self.angle_spin_list.append(QDoubleSpinBox())
            self.angle_spin_list[-1].setMinimum(-180)
            self.angle_spin_list[-1].setMaximum(180)
            self.angle_spin_list[-1].setSingleStep(0.01)
            layout_set_angle.addWidget(self.angle_spin_list[-1], i, 1)
        self.btn_apply_angle = QPushButton("Apply")

        layout_set_angle.addWidget(self.btn_apply_angle, 6, 1)

        angle_set_box.setLayout(layout_set_angle)
        angle_set_box.setMaximumHeight(240)

        xyz_set_box = QGroupBox("XYZ-ABC Set")
        layout_set_angle = QGridLayout()
        self.xyz_spin_list = []
        label_list  = "XYZABC"
        # label_list =  ["X", "Y", "Z", "A", "B", "C", "R_x", "R_y", "R_z"]
        for idx, label in enumerate(label_list):
            layout_set_angle.addWidget(QLabel(label), idx, 0)
            self.xyz_spin_list.append(QDoubleSpinBox())
            self.xyz_spin_list[-1].setMinimum(-180)
            self.xyz_spin_list[-1].setMaximum(180)
            self.xyz_spin_list[-1].setSingleStep(0.01)
            layout_set_angle.addWidget(self.xyz_spin_list[-1], idx, 1)
        self.btn_apply_xyz = QPushButton("Apply")
        self.btn_apply_xyz.clicked.connect(self.on_apply_xyz)
        layout_set_angle.addWidget(self.btn_apply_xyz, len(label_list)+1, 1)
        xyz_set_box.setLayout(layout_set_angle)
        xyz_set_box.setMaximumHeight(240)

        Rot_set_box = QGroupBox("XYZ-Rot Set")
        layout_Rot_angle = QGridLayout()
        self.Rot_spin_list = []
        # label_list  = "XYZABC"
        label_list =  ["X", "Y", "Z", "R_x", "R_y", "R_z"]
        for idx, label in enumerate(label_list):
            layout_Rot_angle.addWidget(QLabel(label), idx, 0)
            self.Rot_spin_list.append(QDoubleSpinBox())
            self.Rot_spin_list[-1].setMinimum(-180)
            self.Rot_spin_list[-1].setMaximum(180)
            self.Rot_spin_list[-1].setSingleStep(0.01)
            layout_Rot_angle.addWidget(self.Rot_spin_list[-1], idx, 1)
        self.btn_apply_Rot = QPushButton("Apply")
        self.btn_apply_Rot.clicked.connect(self.on_apply_Rot)
        layout_Rot_angle.addWidget(self.btn_apply_Rot, len(label_list), 1)
        Rot_set_box.setLayout(layout_Rot_angle)
        Rot_set_box.setMaximumHeight(240)


        container = QVBoxLayout()
        container.addWidget(ip_box)
        container.addWidget(preset_box)
        # container.addWidget(angle_set_box)
        container.addWidget(control_box)
        container.addWidget(xyz_set_box)
        container.addWidget(Rot_set_box)
        # container.addWidget(state_box)


        container.addStretch(2)
        self.setLayout(container)

    def on_connect(self):
        self.client = Client(parent=self)
        self.client.run(block=False)

    def set_msg_rcv(self, msg):
        self.msg_rcv = msg
        # print('listen: ', self.msg_rcv)

        # try:
        for msg_rcv_slice in self.msg_rcv.split(';'):
            # print('msg_rcv_slice:', msg_rcv_slice)
            head = msg_rcv_slice.split(": ")[0].split('cur ')[-1]
            # print(head)
            if head=="angles":
                msg_angle = msg_rcv_slice.split(': ')[-1].split(' ')
                if len(msg_angle)>=6:

                    self.up_ctrl.angle_cur = [float(i) for i in msg_angle[:6]]

                    # self.print_angles_cur()
                    self.up_ctrl.update(cur=True)

                    if self.data_agnle_init==False:
                        # for i in range(6):
                        #     self.spinctrl_angle_list[i].SetValue(self.up_ctrl.angle_cur[i])
                        self.up_ctrl.angles_tgt = self.up_ctrl.angle_cur
                        self.up_ctrl.update(tgt=True)
                        self.data_agnle_init = True

            elif head=="xyz":
                msg_xyz = msg_rcv_slice.split(': ')[-1].split(' ')
                if len(msg_xyz)>=6:

                    self.up_ctrl.xyz_cur = [float(i) for i in msg_xyz[:6]]
                    self.up_ctrl.axis.set_axis(self.up_ctrl.xyz_cur[:3], self.up_ctrl.xyz_cur[3:])

                    if self.data_xyz_init==False:
                        self.up_ctrl.xyz_tgt = self.up_ctrl.xyz_cur
                        self.up_ctrl.update(tgt=True)
                        self.data_xyz_init = True

            elif head=="ft":
                msg_ft = msg_rcv_slice.split(': ')[-1].split(' ')
                if len(msg_ft)>=6:

                    self.up_ctrl.ft_cur = [float(i) for i in msg_ft[:6]]

                    self.up_ctrl.ft.set_data(self.up_ctrl.ft_cur, self.up_ctrl.xyz_cur)

                    # self.ft_traj.append(self.up_ctrl.ft.data)
                    #
                    # if len(self.ft_traj)%50==0:
                    #     print('save')
                    #     np.save('ft_traj', np.array(self.ft_traj))


    def update_tgt(self):

        zero = self.up_ctrl.pre_action_dict['back zero'].copy()
        print('in update tgt: ', self.up_ctrl.xyz_tgt)
        for i in range(6):
            self.xyz_spin_list[i].setValue(self.up_ctrl.xyz_tgt[i])

        for i in range(3):
            # self.Rot_spin_list[i].setValue(self.up_ctrl.xyz_tgt[i]-zero[i])
            self.Rot_spin_list[i].setValue(self.up_ctrl.xyz_tgt[i])

        self.Rot_spin_list[3].setValue(-(self.up_ctrl.xyz_tgt[5] - zero[5]))
        self.Rot_spin_list[4].setValue(-(self.up_ctrl.xyz_tgt[4] - zero[4]))
        self.Rot_spin_list[5].setValue((self.up_ctrl.xyz_tgt[3] - zero[3]))


        # self.update()

        print('connet widget update tgt down')

    def get_tgt_xyz_rot(self):
        # for i in range(3):
        #     pos_set_new[i] = pos_set[i]
        #
        # pos_set_new[3] = pos_set[5] + zero[3]
        # pos_set_new[4] = -pos_set[4] + zero[4]
        # pos_set_new[5] = -pos_set[3] + zero[5]
        #
        # self.up_ctrl.xyz_tgt = pos_set_new
        zero = self.up_ctrl.pre_action_dict['back zero']
        pos_set_new = np.zeros(6)
        pos_set = self.up_ctrl.xyz_tgt
        for i in range(3):
            pos_set_new[i] = pos_set[i]

        # self.Rot_spin_list[3].setValue(-(self.up_ctrl.xyz_tgt[5] - zero[5]))
        # self.Rot_spin_list[4].setValue(-(self.up_ctrl.xyz_tgt[4] - zero[4]))
        # self.Rot_spin_list[5].setValue((self.up_ctrl.xyz_tgt[3] - zero[3]))
        pos_set_new[3] = -pos_set[5] + zero[5]
        pos_set_new[4] = -pos_set[4] + zero[4]
        pos_set_new[5] = pos_set[3] - zero[3]


        return pos_set_new


        # return np.array([self.Rot_spin_list[i].value() for i in range(6)])

    def print_ft_cur(self):
        # pass
        print('print ft cur:', self.up_ctrl.ft_cur)

    def on_mode_radio_btn(self, b):
        # print('on_mode_radio_btn', b.text())
        self.client.write("02,0")
        time.sleep(0.5)
        if b.text()=='Follow':
            if b.isChecked():
                print('check Follow')
                self.client.write("00,FL")
                time.sleep(1.5)
        elif b.text()=='FT Sensor':
            if b.isChecked():
                self.client.write("00,FT_test")
                print('check FT Sensor')

    def on_align_xyz(self):
        self.up_ctrl.xyz_tgt = self.up_ctrl.xyz_cur
        self.up_ctrl.update(tgt=True)

    def on_back_zero(self):
        self.up_ctrl.xyz_tgt = [0.44, 0.04, 0.8, 3.14159, 0, 3.14159]
        self.up_ctrl.angles_tgt = [0, 0, 0, 0, 0, 0]

        self.write_tgt()
        self.up_ctrl.update(tgt=True)

    # Drag Compliance
    def on_force_radio_btn(self, b):
        b_text = b.text()
        # if =='Drag':
        #     self.up_ctrl.drag_force_flag = True
        # elif b.text()=='Compliance':
        #     self.up_ctrl.drag_force_flag = False
        for k in (self.up_ctrl.force_flag_dict):
            # print(k)
            if k==b_text and b.isChecked():
                self.up_ctrl.force_flag_dict[k] = True
            else:
                self.up_ctrl.force_flag_dict[k] = False

        if b_text=='Compliance' and b.isChecked():
            self.up_ctrl.compliance_fa.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()

        print('force_flag_dict', self.up_ctrl.force_flag_dict)

    # def on_check_drag_force(self):
    #     if self.check_drag_force.checkState()==0:
    #         self.up_ctrl.adapt_force_flag = False
    #     else:
    #         self.up_ctrl.adapt_force_flag = True
    #     print('on_check_adapt_force', self.up_ctrl.adapt_force_flag)

    def on_apply_angle(self):
        for i in range(6):
            self.up_ctrl.angles_tgt[i] = (self.spinctrl_angle_list[i]).GetValue()

        print("on_apply_angle")
        print(self.up_ctrl.angles_tgt)
        # print()

    def on_apply_preset(self):
        # print('on preset')
        # print(self.action_combo.currentText())

        self.up_ctrl.xyz_tgt = self.up_ctrl.pre_action_dict[self.action_combo.currentText()].copy()
        # print('on preset set xyz tgt: ', self.up_ctrl.xyz_tgt)
        # self.draw()
        self.up_ctrl.update(tgt=True, sent_tgt=True)

    def on_apply_xyz(self):
        # print("on_apply_xyz")
        self.apply_xyz()

    def apply_xyz(self):
        for i in range(6):
            self.up_ctrl.xyz_tgt[i] = self.xyz_spin_list[i].value()

        self.up_ctrl.axis.set_axis(self.up_ctrl.xyz_tgt[:3], self.up_ctrl.xyz_tgt[3:])
        self.write_tgt()

    def on_apply_Rot(self, pos_set=False):

        pos_set = np.array([self.Rot_spin_list[i].value() for i in range(6)])
        self.apply_Rot(pos_set)

    def apply_Rot(self, pos_set):
        print('before: ', pos_set)
        # cartesian space safe check
        for i in range(6):
            if i<3:
                # pos_set[i] = np.clip(pos_set[i],
                #                      self.up_ctrl.cartesian_work_space[i][0]-self.up_ctrl.pre_action_dict['back zero'][i],
                #                      self.up_ctrl.cartesian_work_space[i][1]-self.up_ctrl.pre_action_dict['back zero'][i])
                pos_set[i] = np.clip(pos_set[i],
                                     self.up_ctrl.cartesian_work_space[i][0],
                                     self.up_ctrl.cartesian_work_space[i][1])
            else:
                pos_set[i] = np.clip(pos_set[i],
                                     self.up_ctrl.cartesian_work_space[i][0],
                                     self.up_ctrl.cartesian_work_space[i][1])
        print('after: ', pos_set)

        zero = self.up_ctrl.pre_action_dict['back zero'].copy()
        pos_set_new = np.zeros(6)
        # print(pos_set, zero)

        for i in range(3):
            # pos_set_new[i] = ( pos_set[i] + zero[i] )
            pos_set_new[i] = pos_set[i]

        pos_set_new[3] = pos_set[5] + zero[3]
        pos_set_new[4] = -pos_set[4] + zero[4]
        pos_set_new[5] = -pos_set[3] + zero[5]

        self.up_ctrl.xyz_tgt = pos_set_new
        self.up_ctrl.axis.set_axis(pos_set_new[:3], pos_set_new[3:])
        self.up_ctrl.ft.set_data(self.up_ctrl.ft_cur, pos_set_new)
        self.write_tgt()

    def write_tgt(self):

        # print("write_tgt: ", self.up_ctrl.xyz_tgt)
        str_tmp = '01,{'
        for i in range(6):
            str_tmp += str(self.up_ctrl.xyz_tgt[i])
            if i<5: str_tmp += ','
        str_tmp += '};'

        if self.client is not None:
            # pass
            self.client.write(str_tmp)
        else:
            self.up_ctrl.xyz_cur = self.up_ctrl.xyz_tgt

        self.up_ctrl.update(tgt=True)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_tgt = [0]*6
        self.angle_tgt = [0]*6
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6

        self.connect_widget = ConnectWidget(up_ctrl=self)
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