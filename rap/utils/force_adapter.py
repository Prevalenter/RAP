import time

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore import QTimer
from threading import Timer, Thread

class DragForceAdapter:
    def __init__(self, up_ctrl=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.dt = dt

        self.timer = QTimer(self.up_ctrl)
        self.timer.timeout.connect(self.timer_step)
        self.timer.start(int(self.dt*1000))

    def timer_step(self):
        if self.up_ctrl is not None and self.up_ctrl.force_flag_dict['Drag']:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])
            if force_contact_norm!=0:

                xyz_rot_cur = self.up_ctrl.connect_widget.get_tgt_xyz_rot()

                dx = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)

                dx[:3] = np.clip(force_contact_world[:3], -10, 10)*0.01

                torque_map = np.zeros(3)
                torque_map[0] = force_contact_world[4]
                torque_map[1] = force_contact_world[3]
                torque_map[2] = -force_contact_world[5]

                dx[3:] = torque_map

                xyz_rot_new = xyz_rot_cur.copy()
                xyz_rot_new += dx

                self.up_ctrl.connect_widget.apply_Rot(xyz_rot_new)




class ComplianceForceAdapter:
    def __init__(self, up_ctrl=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.dt = dt

        self.timer = QTimer(self.up_ctrl)
        self.timer.timeout.connect(self.timer_step)
        self.timer.start(int(self.dt*1000))

        # M (\ddX_r-\ddX) + B (\dx_r - \dx) + K (x_r - x) = F - F_d
        self.M = np.array([0, 0, 0, 0, 0, 0])
        self.B = np.array([0, 0, 0, 0, 0, 0])
        self.K = np.array([10, 10, 10, 10000, 10000, 10000])

        self.ddx_r = np.array([0, 0, 0, 0, 0, 0])
        self.dx_r = np.array([0, 0, 0, 0, 0, 0])
        self.x_r = np.array([0, 0, 0, 0, 0, 0])

        self.ddx = np.array([0, 0, 0, 0, 0, 0])
        self.dx = np.array([0, 0, 0, 0, 0, 0])

        self.F_d = np.array([0, 0, 0, 0, 0, 0])

        self.pos_cur = np.zeros(6)
        self.pos_cur_last = np.zeros(6)
        self.pos_cur_last2 = np.zeros(6)

    def timer_step(self):
        if self.up_ctrl is not None and self.up_ctrl.force_flag_dict['Compliance']:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            # force_contact_world = np.array([1, 0, 0, 0, 0, 0])
            force_contact_norm = np.linalg.norm(force_contact_world[:3])

            self.dx = (self.pos_cur-self.pos_cur_last)/self.dt
            dx_last = (self.pos_cur_last-self.pos_cur_last2)/self.dt
            self.ddx = (self.dx-dx_last)/self.dt

            if force_contact_norm!=0:
                print(self.x_r)
                # xyz_rot_cur = self.up_ctrl.connect_widget.get_tgt_xyz_rot()
                #

                # dx[:3] = np.clip(force_contact_world[:3], -10, 10)*0.01

                # torque_map = np.zeros(3)
                # torque_map[0] = force_contact_world[4]
                # torque_map[1] = force_contact_world[3]
                # torque_map[2] = -force_contact_world[5]

                # dx[3:] = torque_map
                F = force_contact_world

                dx = -( F - self.F_d - self.M * (self.ddx_r-self.ddx) - self.B * (self.dx_r - self.dx) )/self.K
                dx = -dx
                print(dx)

                xyz_rot_new = self.x_r.copy()
                xyz_rot_new += dx
                #
                self.up_ctrl.connect_widget.apply_Rot(xyz_rot_new)
            else:
                self.up_ctrl.connect_widget.apply_Rot(self.x_r)





if __name__ == '__main__':
    dt = 0.1

    fa = DragForceAdapter()

    while(True):
        pass

    # x = np.zeros(6)
    #
    # x_list = []
    # for i in range(100):
    #     if i<10 or (i>60 and i<70):
    #         force = np.array([1, 1, 1, 0, 0, 0])*0.5
    #     else:
    #         force = np.zeros(6)
    #     x += fa.step(force)
    #     x_list.append(x.copy())
    #     print(x)
    #
    # x_list = np.array(x_list)
    # print(x_list.shape)
    #
    # for i in range(6):
    #     plt.subplot(6, 1, 1+i)
    #
    #     plt.plot(x_list[:, i])
    #
    # plt.show()
