import numpy as np

from PyQt5.QtCore import QTimer

class PegInHoleControl:
    def __init__(self, up_ctrl=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.dt = dt
        self.assemble_stage_flage = 0

        if up_ctrl is not None:
            self.timer = QTimer(self.up_ctrl)
            self.timer.timeout.connect(self.run)
            # self.timer.start(int(self.dt*1000))

        self.xy_tgt = np.array( [0.4776, 0.3969] )

        self.F_r = np.array([0, 0, 5, 0, 0, 0])
        self.P = np.array([0, 0, 1e-4, 0, 0, 0])
        self.I = np.array([0, 0, 0, 0, 0, 0])
        self.D = np.array([0, 0, 0, 0, 0, 0])

        self.error_sum = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.error_last = np.array([0, 0, 0, 0, 0, 0])

    def set_label_ctrl_state(self, label_ctrl_state):
        self.label_ctrl_state = label_ctrl_state

    # def timer_step(self):
    def run(self):
        print('PegInHoleControl run')

        is_contraol = True
        if self.up_ctrl is not None and is_contraol:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])
            torque_norm = np.linalg.norm(force_contact_world[3:])
            # force_contact_norm = np.linalg.norm(force_contact_world[:3])

            if force_contact_norm>40 or torque_norm>0.5:
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

                dx = np.zeros(6)

                # no contact
                if self.assemble_stage_flage==0:
                    if force_contact_norm==0: # no conatact
                        dx = np.array([0, 0, -8e-5, 0, 0, 0])
                    else:
                        self.assemble_stage_flage = 1
                        self.label_ctrl_state.setText(f"Control: stage 1")

                if self.assemble_stage_flage==1: # reach the force desired
                    dx = np.zeros(6)
                    dx[2] = 5e-6*np.sign(error[2])

                    if np.abs(error[2]) < 1:
                        self.assemble_stage_flage = 2
                        self.label_ctrl_state.setText(f"Control: stage 2")

                if self.assemble_stage_flage==2: # to the hole
                    dx = np.zeros(6)

                    dx[:2] = -(8e-5)*np.sign(self.x_r[:2]-self.xy_tgt[:2])

                    # keep the force in z axis constant
                    # dx[2] = 5e-6 * np.sign(error[2])
                    if np.abs(error[2])>0.5:
                        dx[2] = 5e-6 * np.sign(error[2])

                    if self.x_r[2] < 0.172 or force_contact_norm<2:
                        self.assemble_stage_flage = 3
                        self.label_ctrl_state.setText(f"Control: stage 3")

                if self.assemble_stage_flage==3:
                    dx = np.zeros(6)
                    if np.abs(error[0])>0.5:
                        dx[0] = 2e-6 * np.sign(error[0])
                    if np.abs(error[0])>0.5:
                        dx[1] = 2e-6 * np.sign(error[1])

                    if np.abs(error[2])>0.5:
                        dx[2] = 5e-5 * np.sign(error[2])

                    if np.abs(error[3]) > 0.2:
                        dx[3] = 1e-4 * np.sign(error[3])

                    if np.abs(error[4]) > 0.2:
                        dx[4] = 1e-4 * np.sign(error[4])

                    if self.x_r[2] < 0.120:
                        self.assemble_stage_flage = 4
                        self.label_ctrl_state.setText(f"Control: stage 4")

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











class PegInHoleControl:
    def __init__(self, up_ctrl=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.dt = dt
        self.assemble_stage_flage = 0

        if up_ctrl is not None:
            self.timer = QTimer(self.up_ctrl)
            self.timer.timeout.connect(self.run)
            # self.timer.start(int(self.dt*1000))

        self.xy_tgt = np.array( [0.4776, 0.3969] )

        self.F_r = np.array([0, 0, 5, 0, 0, 0])
        self.P = np.array([0, 0, 1e-4, 0, 0, 0])
        self.I = np.array([0, 0, 0, 0, 0, 0])
        self.D = np.array([0, 0, 0, 0, 0, 0])

        self.error_sum = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.error_last = np.array([0, 0, 0, 0, 0, 0])


        # diffusion policy deploy

    def set_label_ctrl_state(self, label_ctrl_state):
        self.label_ctrl_state = label_ctrl_state

    # def timer_step(self):
    def run(self):
        print('PegInHoleControl run')

        is_contraol = True
        if self.up_ctrl is not None and is_contraol:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])
            torque_norm = np.linalg.norm(force_contact_world[3:])
            # force_contact_norm = np.linalg.norm(force_contact_world[:3])

            if force_contact_norm>40 or torque_norm>0.5:
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

                dx = np.zeros(6)

                # no contact
                if self.assemble_stage_flage==0:
                    if force_contact_norm==0: # no conatact
                        dx = np.array([0, 0, -8e-5, 0, 0, 0])
                    else:
                        self.assemble_stage_flage = 1
                        self.label_ctrl_state.setText(f"Control: stage 1")

                if self.assemble_stage_flage==1: # reach the force desired
                    dx = np.zeros(6)
                    dx[2] = 5e-6*np.sign(error[2])

                    if np.abs(error[2]) < 1:
                        self.assemble_stage_flage = 2
                        self.label_ctrl_state.setText(f"Control: stage 2")

                if self.assemble_stage_flage==2: # to the hole
                    dx = np.zeros(6)

                    dx[:2] = -(8e-5)*np.sign(self.x_r[:2]-self.xy_tgt[:2])

                    # keep the force in z axis constant
                    # dx[2] = 5e-6 * np.sign(error[2])
                    if np.abs(error[2])>0.5:
                        dx[2] = 5e-6 * np.sign(error[2])

                    if self.x_r[2] < 0.172 or force_contact_norm<2:
                        self.assemble_stage_flage = 3
                        self.label_ctrl_state.setText(f"Control: stage 3")

                if self.assemble_stage_flage==3:
                    dx = np.zeros(6)
                    if np.abs(error[0])>0.5:
                        dx[0] = 2e-6 * np.sign(error[0])
                    if np.abs(error[0])>0.5:
                        dx[1] = 2e-6 * np.sign(error[1])

                    if np.abs(error[2])>0.5:
                        dx[2] = 5e-5 * np.sign(error[2])

                    if np.abs(error[3]) > 0.2:
                        dx[3] = 1e-4 * np.sign(error[3])

                    if np.abs(error[4]) > 0.2:
                        dx[4] = 1e-4 * np.sign(error[4])

                    if self.x_r[2] < 0.120:
                        self.assemble_stage_flage = 4
                        self.label_ctrl_state.setText(f"Control: stage 4")

                self.error_last = error

                self.x_r += dx
                #
                self.up_ctrl.connect_widget.apply_Rot(self.x_r)
            else:
                self.up_ctrl.connect_widget.apply_Rot(self.x_r)

