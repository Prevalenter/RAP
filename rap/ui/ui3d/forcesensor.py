import numpy as np
from scipy.spatial.transform import Rotation as R
import moderngl

import sys
sys.path.append("../..")
from utils.trans import get_transform, to_world_axis, ft_map

from utils.ft_ident import get_ft_para, get_force_contact_once, get_torque_contact_once

class ForceSensor:
    def __init__(self, scale=1):
        self.ft_sensor = np.zeros(6)
        self.pos_cur = np.zeros(6)
        self.ft_contact = np.zeros(6)

        self.force_contact_world = np.zeros(6)

        self.initial_flag = False
        self.initial_value = np.zeros(6)
        self.initial_pos = None

        self.scale = scale
        self.axis_base = np.array([0, 0, 0, 1])[:, None]

        self.load_ft_para()

    def load_ft_para(self):
        self.para_force, self.para_torque = get_ft_para()


    def set_data(self, data, pos):
        self.ft_sensor = np.array(data)
        self.pos_cur = np.array(pos)

        if np.linalg.norm(self.ft_sensor)==0:
            self.ft_sensor = np.zeros(6)
        else:
            self.ft_contact[:3] = get_force_contact_once(self.para_force, self.ft_sensor, self.pos_cur)[0]
            self.ft_contact[3:] = get_torque_contact_once(self.para_torque, self.ft_sensor)[0]
            if np.linalg.norm(self.ft_contact[:3])<0.2:
                self.ft_contact[:3] = 0

            if np.linalg.norm(self.ft_contact[3:])<0.005:
                self.ft_contact[3:] = 0

            self.set_ft_world()

    def set_ft_world(self):
        # T = self.calucu()
        T = to_world_axis(self.pos_cur[:3], self.pos_cur[3:])
        begin = T @ self.axis_base
        self.begin = begin[:3, 0]

        end = T @ np.array(list(self.ft_contact[:3]) + [1])[:, None]
        self.end = end[:3, 0]
        print(end.shape, begin.shape, self.force_contact_world[:3].shape)
        self.force_contact_world[:3] = self.end - self.begin

        # self.force_contact_world[3:] = self.ft_contact[3:]
        self.force_contact_world[3] = self.ft_contact[4]
        self.force_contact_world[4] = self.ft_contact[3]
        self.force_contact_world[5] = -self.ft_contact[5]

    def render(self, gl_ctrl):
        self.set_ft_world()

        gl_ctrl.color.value = gl_ctrl.new_color
        gl_ctrl.vbo_ft = gl_ctrl.ctx.buffer(np.array([self.begin, self.end], dtype=np.float32))
        gl_ctrl.vao_ft = gl_ctrl.ctx.simple_vertex_array(gl_ctrl.prog, gl_ctrl.vbo_ft, 'in_position')

        # Render reigon loop
        # tmp[tmp==0] = 0.2
        gl_ctrl.color.value = (1, 1, 1, gl_ctrl.grid_alpha_value)
        gl_ctrl.vao_ft.render(moderngl.LINES)


    def transform(self, axis='global'):
        # axis: global tool sensor
        pass

    # def set_compensate(self, s):
    #     self.compensate = s


if __name__ == '__main__':
    a = ForceSensor()
    a.render(None)

