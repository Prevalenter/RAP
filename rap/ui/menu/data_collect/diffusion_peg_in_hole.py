
import numpy as np
import time
import sys
sys.path.append('../../../')

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pandas as pd
import datetime
from pathlib import Path

from ui.base.vedio import VedioWidget
from ui.base.state import StateWidget
from utils.realsense import SingleReansense

from ui.menu.data_collect.peg_in_hole_control import PegInHoleControl
from ui.menu.data_collect.auto_sample import AutoSampleThread


import cv2

# size = 720*16//9, 720
# duration = 2
# fps = 25
def np2img(rgbd, save_dir):
    # t, cam, w, h, c = rgbd.shape
    for cam_idx in range(rgbd.shape[1]):
        rgbd_cam = rgbd[:, cam_idx]
        # t = time.time()
        for i in range(rgbd_cam.shape[0]):
            img = rgbd_cam[i].copy().astype(np.float64)
            img[3] /= 65535
            print('save:', f"{save_dir}/img/{i}_cam{cam_idx}.png")
            cv2.imwrite(f"{save_dir}/img/{i}_cam{cam_idx}.png", img)


class DiffusionPegInHoleWidget(QDialog):
    def __init__(self, single_cam_list, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl
        self.auto_sample = AutoSampleThread(self)

        self.step_index = 0
        self.save_mode = 'manual'

        self.single_cam_list = single_cam_list

        self.peg_in_hole_ctrl = PegInHoleControl(up_ctrl=up_ctrl)

        self.data_init()
        self.ui_init()

    def data_init(self):
        self.data = {
            "idx": 0,
            "img": [],
            "force_torque": [],
            "xyz_rot": [],
            'assemble_stage':[]
        }

    def ui_init(self):
        container = QVBoxLayout()

        self.resize(650, 800)

        self.label_cam_state = QLabel("")

        self.label_ctrl_state = QLabel("Control: ")
        self.peg_in_hole_ctrl.set_label_ctrl_state(self.label_ctrl_state)

        self.btn_initial_position = QPushButton("Initial Position")
        self.btn_random_position = QPushButton("Random Position")
        self.btn_run_assemble = QPushButton("Run Assemble")
        self.btn_stop_assemble = QPushButton("Stop Assemble")
        self.btn_out_hole = QPushButton("out the hole")


        self.btn_initial_position.clicked.connect(self.on_initial_position)
        self.btn_random_position.clicked.connect(self.on_random_position)
        self.btn_run_assemble.clicked.connect(self.on_run_assemble)
        self.btn_stop_assemble.clicked.connect(self.on_stop_assemble)
        self.btn_out_hole.clicked.connect(self.on_out_hole)

        # self.ctrl_layout = QHBoxLayout()
        self.ctrl_layout = QGridLayout()

        self.ctrl_layout.addWidget(self.label_ctrl_state, 0, 0)
        self.ctrl_layout.addWidget(self.btn_initial_position, 0, 1)
        self.ctrl_layout.addWidget(self.btn_random_position, 0, 2)
        self.ctrl_layout.addWidget(self.btn_run_assemble, 0, 3)
        self.ctrl_layout.addWidget(self.btn_stop_assemble, 0, 4)
        self.ctrl_layout.addWidget(self.btn_out_hole, 0, 5)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save")

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_save.clicked.connect(self.on_save)

        # self.ctrl_layout.addWidget(self.label_state)
        # self.ctrl_layout.addWidget(self.btn_start)
        # self.ctrl_layout.addWidget(self.btn_stop)
        # self.ctrl_layout.addWidget(self.btn_clear)
        # self.ctrl_layout.addWidget(self.btn_save)

        self.ctrl_layout.addWidget(self.label_cam_state, 1, 0)
        self.ctrl_layout.addWidget(self.btn_start, 1, 1)
        self.ctrl_layout.addWidget(self.btn_stop, 1, 2)
        self.ctrl_layout.addWidget(self.btn_clear, 1, 3)
        self.ctrl_layout.addWidget(self.btn_save, 1, 4)


        self.sample_process = QLabel("")
        self.btn_auto_start = QPushButton("Auto Start")
        self.btn_auto_stop = QPushButton("Auto Stop")
        self.label_ctrl_auto = QLabel("Auto: ")
        self.ctrl_layout.addWidget(self.label_ctrl_auto, 2, 0)
        # self.ctrl_layout.addWidget(self.sample_process, 2, 1)
        self.ctrl_layout.addWidget(self.btn_auto_start, 2, 1)
        self.ctrl_layout.addWidget(self.btn_auto_stop, 2, 2)
        self.btn_auto_start.clicked.connect(self.on_auto_start)
        self.btn_auto_stop.clicked.connect(self.on_auto_stop)

        self.auto_sample.start_run.connect(self.on_run_assemble)
        self.auto_sample.stop_run.connect(self.on_stop_assemble)

        self.auto_sample.start_run.connect(self.on_start)
        self.auto_sample.stop_run.connect(self.on_stop)

        self.state_widget = StateWidget()

        container.addLayout(self.ctrl_layout)
        container.addWidget(self.state_widget)

        self.cam_list = []
        for idx in range(len(self.single_cam_list)):
            cam = VedioWidget(single_cam=self.single_cam_list[idx])
            cam.setParent(self)
            self.cam_list.append(cam)

            container.addWidget(cam)
        container.addStretch(2)
        self.setLayout(container)

        # self.statusbar = QStatusBar()
        # self.setStatusTip("1231")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)

        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self.get_step_data)
          # every 10,000 milliseconds


    def on_auto_start(self):
        self.save_mode = 'auto'
        self.auto_save_path = '0001'

        self.auto_sample_flag = True
        self.auto_sample.start()
        # self.label_ctrl_auto.setText(f"Auto:    0/200")

    def on_auto_stop(self):
        self.save_mode = 'manual'

        self.auto_sample_flag = False
        # self.auto_sample.stop()
        # self.label_ctrl_auto.setText(f"Auto:    0/200")

    def on_initial_position(self):
        print("on_initial_position")
        self.peg_in_hole_ctrl.timer.stop()

        self.up_ctrl.connect_widget.apply_Rot( np.array([0.4776, 0.3969, 0.2, 0, 0, 0]) )

    def on_random_position(self):
        print("on_random_position")
        self.peg_in_hole_ctrl.timer.stop()

        # self.up_ctrl.connect_widget.apply_Rot(np.array([0.4500, 0.3969, 0.2, 0, 0, 0]))

        r_min, r_max = 0.007, 0.015
        r = np.random.rand(1) * (r_max - r_min) + r_min
        angle = np.random.rand(1) * 2 * np.pi

        x = r * np.cos(angle) + self.peg_in_hole_ctrl.xy_tgt[0]
        y = r * np.sin(angle) + self.peg_in_hole_ctrl.xy_tgt[1]

        random_pos = np.array([x, y, 0.18, 0, 0, 0]).astype(np.float64)
        # print(random_pos)

        self.up_ctrl.connect_widget.apply_Rot( random_pos )

    def on_run_assemble(self):
        print("on_run_assemble")

        self.peg_in_hole_ctrl.assemble_stage_flage = 0

        self.peg_in_hole_ctrl.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()
        self.peg_in_hole_ctrl.timer.start(int(self.peg_in_hole_ctrl.dt*1000))

        self.label_ctrl_state.setText(f"Control: stage {self.peg_in_hole_ctrl.assemble_stage_flage}")

    def on_stop_assemble(self):
        print("on stop assemble")
        self.peg_in_hole_ctrl.timer.stop()

        x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()
        self.up_ctrl.connect_widget.apply_Rot(x_r)

        self.label_ctrl_state.setText(f"Control: stop")

    def out_hole(self):
        x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()
        x_r[2] += 0.01
        self.up_ctrl.connect_widget.apply_Rot(x_r)
        if x_r[2] > 0.2:
            self.out_hole_timer.stop()
            self.label_ctrl_state.setText(f"Control: ")

    def on_out_hole(self):
        self.peg_in_hole_ctrl.timer.stop()
        self.out_hole_timer = QTimer()
        self.out_hole_timer.timeout.connect( self.out_hole )
        self.out_hole_timer.start(1000)

        self.label_ctrl_state.setText(f"Control: out hole")

    # img save as png, zarr
    def on_save(self, img_save_as='png', exp_dir=None):
        self.on_stop()

        if exp_dir is None:
            '''CREATE DIR'''
            timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
            exp_dir = Path('../data/diffusion_peg_in_hole/')
            exp_dir.mkdir(exist_ok=True)
            # exp_dir = exp_dir.joinpath(args.model)
            # exp_dir.mkdir(exist_ok=True)
            exp_dir = exp_dir.joinpath(timestr)
            exp_dir.mkdir(exist_ok=True)

            exp_img_dir = exp_dir.joinpath('img')
            exp_img_dir.mkdir(exist_ok=True)


        print("save", exp_dir)

        img_data = np.array(self.data['img'])

        t = time.time()
        if img_save_as=='png':
            np2img(img_data, exp_dir)
        else:
            # np2zarr()
            pass
        print('save img using:', time.time()-t)

        # save ft and xyz data
        ft = np.array(self.data['force_torque']).astype(np.float32)
        xyz = np.array(self.data['xyz_rot']).astype(np.float32)
        # print(ft.shape, xyz.shape)
        df_ft = pd.DataFrame(ft)
        # print(ft)
        df_ft.to_csv(exp_dir.joinpath("ft.csv"), header=False, index=False)

        df_xyz = pd.DataFrame(xyz)
        df_xyz.to_csv(exp_dir.joinpath("xyz.csv"), header=False, index=False)


    def on_start(self):
        print('on start')
        self.data_timer.start(100)

    def on_stop(self):
        print("on stop")
        self.data_timer.stop()

    def on_clear(self):
        self.data_init()

    def get_step_data(self):
        # QLabel("")
        self.data["idx"] += 1

        # self.data = {
        #     "idx": 0,
        #     "img": [],
        #     "force_torque": [],
        #     "xyz_rot": []
        # }
        if self.up_ctrl is not None:
            self.data["img"].append( np.array([ cam.th.single_cam.rgbd for cam in self.cam_list]) )
            self.data['force_torque'].append(np.array(self.up_ctrl.ft.force_contact_world))
            self.data['xyz_rot'].append(np.array(self.up_ctrl.xyz_cur))
            # self.peg_in_hole_ctrl.assemble_stage_flage
            self.data['assemble_stage'].append(self.peg_in_hole_ctrl.assemble_stage_flage)
        else:
            self.data["img"].append( np.array([ cam.th.single_cam.rgbd for cam in self.cam_list]) )
            self.data['force_torque'].append( np.zeros(6) )
            self.data['xyz_rot'].append( np.zeros(6) )

        # print(self.data['img'][-1].shape)


    def update(self):
        # print('diffusion peg in hole update')

        idx = self.data["idx"]
        self.label_cam_state.setText(f"Camera:    length of data: {idx}")

        if self.up_ctrl is not None:

            self.state_widget.update(self.up_ctrl.angle_cur,
                                     self.up_ctrl.xyz_cur,
                                     self.up_ctrl.ft.ft_sensor,
                                     self.up_ctrl.ft.force_contact_world)


        else:

            self.state_widget.update([0]*6,
                                     [0]*6,
                                     [0]*6,
                                     [0]*6,)


    def showEvent(self, a0):
        print('diffusion show event')
        self.timer.start(200)

    def closeEvent(self, a0):
        print('diffusion close event')
        self.timer.stop()


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0]*6
        self.angle_cur = [0]*6
        self.ft_cur = [0]*6


        self.dev_id_list = ['147122075207', '147322070524']
        self.sing_cam_list = []
        for i in range( len(self.dev_id_list ) ):
            self.sing_cam_list.append( SingleReansense(self.dev_id_list[i]) )

        # self.peg_in_hole_widget = DiffusionPegInHoleWidget(self.sing_cam_list, up_ctrl=self)

        self.peg_in_hole_widget = DiffusionPegInHoleWidget( self.sing_cam_list, up_ctrl=None )
        self.peg_in_hole_widget.setParent(self)

        self.resize(800, 800)


    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.state_widget.update_tgt()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # r_min, r_max = 0.01, 0.03
    # r = np.random.rand(5000)*(r_max-r_min) + r_min
    # angle = np.random.rand(5000)*2*np.pi
    #
    # x = r * np.cos(angle)
    # y = r * np.sin(angle)

    # print(angle)
    # plt.scatter(x, y)
    # plt.show()



    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec_()


