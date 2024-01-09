import numpy as np

from PyQt5.QtCore import QTimer

class PegInHoleContral:
    def __init__(self, up_ctrl=None, dt=0.05):
        self.up_ctrl = up_ctrl
        self.dt = dt

        if up_ctrl is not None:
            self.timer = QTimer(self.up_ctrl)
            self.timer.timeout.connect(self.run)
            # self.timer.start(int(self.dt*1000))

        self.xy_tgt = np.array( [0.4776, 0.3969, 0.1720] )

        self.F_r = np.array([0, 0, 5, 0, 0, 0])
        self.P = np.array([0, 0, 1e-4, 0, 0, 0])
        self.I = np.array([0, 0, 0, 0, 0, 0])
        self.D = np.array([0, 0, 0, 0, 0, 0])

        self.error_sum = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.error_last = np.array([0, 0, 0, 0, 0, 0])

    # def timer_step(self):
    def run(self):
        print('PegInHoleContral run')

        is_contraol = True
        if self.up_ctrl is not None and is_contraol:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])

            if force_contact_norm>40:
                return

            # if force_contact_norm!=0:
            if True:
                print('-'*60)
                print(force_contact_world)

                # only reaction to the force in z positive
                if force_contact_world[2]<1:
                    force_contact_world[2] = 0

                error = force_contact_world - self.F_r
                self.error_sum += error
                # dx = self.P*(error) + self.I*self.error_sum - self.D*(error-self.error_last)

                # flag is 0: peg down, to contact the platform
                # 0 -> 1: the force reach the range
                # flag is 1: slide in the platform, keep the force in z constant
                # 1 -> 2 : the z position less the threshold
                # flag is 2: get the hole, the peg down, and the x and y axis become soft

                # no contact
                if force_contact_norm==0:
                    dx = np.array([0, 0, -1e-4, 0, 0, 0])
                # contact
                else:
                    dx = np.zeros(6)
                    if self.x_r[2] > 0.172:
                        dx[2] = 5e-6*np.sign(error[2])
                    else:
                        dx[2] = 5e-5*np.sign(error[2])

                if np.abs(error[2])<1 and self.x_r[0]<0.4776:
                    dx[0] = (5e-5)

                self.error_last = error

                self.x_r += dx
                #
                self.up_ctrl.connect_widget.apply_Rot(self.x_r)
            else:
                self.up_ctrl.connect_widget.apply_Rot(self.x_r)

    def reaction_foce(self, pos):
        if pos[2]>0.55:
            return np.zeros(6)
        else:
            dx = 0.55-np.round(pos[2], 3)
            # print(dx)
            return np.array([0, 0, np.exp(100*dx)+3, 0, 0, 0])

