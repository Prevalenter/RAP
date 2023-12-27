import time

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore import QTimer
from threading import Timer, Thread

class ForceAdapter:
    def __init__(self, up_ctrl=None, dt=0.1, mass=[3, 3, 3, 4e-3, 4e-3, 4e-3], damp = [0.8, 0.8, 0.8, 0, 0 ,0]):
        self.up_ctrl = up_ctrl
        self.v = np.zeros(6)

        self.set_mass(mass)
        self.set_damp(damp)
        self.set_dt(dt)

        self.timer = QTimer(self.up_ctrl)
        self.timer.timeout.connect(self.timer_step)
        self.timer.start(100)

    def set_dt(self, dt):
        self.dt = dt

    def set_mass(self, mass):
        self.mass = mass

    def set_damp(self, damp):
        self.damp = damp

    def timer_step(self):
        # while True:
            # if self.up_ctrl is not None and self.up_ctrl.connect_widget.client is not None:
        if self.up_ctrl is not None and self.up_ctrl.adapt_force_flag:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])
            if force_contact_norm!=0:
                print('-'*50)
                print('in change xyz rot')
                xyz_rot_cur = self.up_ctrl.connect_widget.get_tgt_xyz_rot()

                print(xyz_rot_cur, force_contact_world)

                dx = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)


                dx[:3] = np.clip(force_contact_world[:3], -10, 10)*0.01

                torque_map = np.zeros(3)
                torque_map[0] = force_contact_world[4]
                torque_map[1] = force_contact_world[3]
                torque_map[2] = -force_contact_world[5]

                dx[3:] = torque_map

                xyz_rot_new = xyz_rot_cur.copy()
                print(xyz_rot_cur.shape, dx)
                xyz_rot_new += dx

                self.up_ctrl.connect_widget.apply_Rot(xyz_rot_new)
                # self.up_ctrl.connect_widget.apply_Rot([0, 0, 0, 0, 0, 0])
                # is_restart = False

                print('out change xyz rot')


    def step(self, force):
        # apply the force-translate
        a = force/self.mass[:3]
        self.v[:3] = a*(self.dt**2) + self.damp[:3]*self.v[:3]

        # apply the torque-rotation

        return self.v*self.dt


if __name__ == '__main__':
    dt = 0.1

    fa = ForceAdapter()

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
