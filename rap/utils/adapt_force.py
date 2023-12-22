import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore import QTimer
from threading import Timer

class ForceAdapter:
    def __init__(self, up_ctrl=None, dt=0.1, mass=[3, 3, 3, 4e-3, 4e-3, 4e-3], damp = [0.8, 0.8, 0.8, 0, 0 ,0]):
        self.up_ctrl = up_ctrl
        self.v = np.zeros(6)

        self.set_mass(mass)
        self.set_damp(damp)
        self.set_dt(dt)

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.timer_step)
        # self.timer.start(100)

        self.timer_step()

    def set_dt(self, dt):
        self.dt = dt

    def set_mass(self, mass):
        self.mass = mass

    def set_damp(self, damp):
        self.damp = damp

    def timer_step(self):
        if self.up_ctrl is not None:
            self.up_ctrl.ft.


        print('timer_step')
        timer = Timer(0.1, self.timer_step)
        timer.start()

    def step(self, force):
        a = force/self.mass
        self.v = a*(self.dt**2) + self.damp*self.v
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
