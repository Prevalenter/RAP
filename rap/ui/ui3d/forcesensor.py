import numpy as np
from scipy.spatial.transform import Rotation as R
import moderngl

import sys
sys.path.append("../..")
from utils.trans import get_transform, to_world_axis, ft_map

from utils.ft_ident import get_ft_para, get_force_conttace_once

class ForceSensor:
    def __init__(self, scale=1):
        self.ft_sensor = np.zeros(6)

        self.pos_cur = np.zeros(6)

        self.ft_contact = np.zeros(6)

        self.initial_flag = False
        self.initial_value = np.zeros(6)
        self.initial_pos = None

        self.scale = scale

        self.axis_base = np.array([0, 0, 0, 1])[:, None]

        self.para = get_ft_para()


    def set_data(self, data, pos):

        self.ft_sensor = np.array(data)
        self.pos_cur = np.array(pos)

        self.ft_contact = get_force_conttace_once(self.para, self.ft_sensor, self.pos_cur)


    def render(self, gl_ctrl):
        # T = self.calucu()
        T = to_world_axis(self.pos_cur[:3], self.pos_cur[3:])
        begin = T @ self.axis_base
        begin = begin[:3, 0]

        end = T @ np.array(list(self.ft_contact[:3]) + [1])[:, None]
        end = end[:3, 0]

        gl_ctrl.color.value = gl_ctrl.new_color
        gl_ctrl.vbo_ft = gl_ctrl.ctx.buffer(np.array([begin, end], dtype=np.float32))
        gl_ctrl.vao_ft =  gl_ctrl.ctx.simple_vertex_array(gl_ctrl.prog, gl_ctrl.vbo_ft, 'in_position')
        # Render reigon loop
        # tmp[tmp==0] = 0.2
        gl_ctrl.color.value = (1, 1, 1, gl_ctrl.grid_alpha_value)
        gl_ctrl.vao_ft.render(moderngl.LINES)

    # get the force in world axis
    # def ft_in_world(self, ft, pos):
    #     pass

    # def calucu(self):
    #     xyz = self.pos_cur[:3].copy()
    #     angles = self.pos_cur[3:].copy()
    #     angles[0] = -angles[0]
    #     angles[2] = -angles[2]
    #
    #     return get_transform(xyz, angles)

    def transform(self, axis='global'):
        # axis: global tool sensor
        pass

    # def set_compensate(self, s):
    #     self.compensate = s


if __name__ == '__main__':
    a = ForceSensor()
    a.render(None)

