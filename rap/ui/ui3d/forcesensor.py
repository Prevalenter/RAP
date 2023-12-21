import numpy as np
from scipy.spatial.transform import Rotation as R
import moderngl

import sys
sys.path.append("../..")
from utils.trans import get_transform

class ForceSensor:
    def __init__(self, scale=1):
        self.data = np.zeros(6)


        self.pos_cur = np.zeros(6)


        self.data_modefied = np.zeros(6)

        self.initial_flag = False
        self.initial_value = np.zeros(6)
        self.initial_pos = None

        # self.data[:3] = 1
        self.scale = scale

        self.axis_base = np.array([0, 0, 0, 1])[:, None]


    def set_data(self, data, pos):
        if not self.initial_flag and np.linalg.norm(data)>1e-1:
            self.initial_value = np.array(data)
            self.initial_pos = pos
            print('force intitial state', self.initial_value)
            self.initial_flag = True

        self.data = np.array(data)
        self.pos_cur = np.array(pos)
        self.data_modefied = self.data - self.initial_value

    def render(self, gl_ctrl):
        T = self.calucu()
        begin = T @ self.axis_base
        begin = begin[:3, 0]
        self.data_modefied[:3] = [0.1, 0.1, 0.1]
        end = T @ np.array(list(self.data_modefied[:3]) + [1])[:, None]
        end = end[:3, 0]
        # print(T)
        # print(begin[:3], end[:3])
        # print('-'*30)

        gl_ctrl.color.value = gl_ctrl.new_color
        gl_ctrl.vbo_ft = gl_ctrl.ctx.buffer(np.array([begin, end], dtype=np.float32))
        gl_ctrl.vao_ft =  gl_ctrl.ctx.simple_vertex_array(gl_ctrl.prog, gl_ctrl.vbo_ft, 'in_position')
        # Render reigon loop
        # tmp[tmp==0] = 0.2
        gl_ctrl.color.value = (1, 1, 1, gl_ctrl.grid_alpha_value)
        gl_ctrl.vao_ft.render(moderngl.LINES)

    # get the force in world axis
    def ft_in_world(self, ft, pos):
        pass

    def calucu(self):
        xyz = self.pos_cur[:3].copy()
        angles = self.pos_cur[3:].copy()
        angles[0] = -angles[0]
        angles[2] = -angles[2]

        return get_transform(xyz, angles)

    def transform(self, axis='global'):
        # axis: global tool sensor
        pass

    # def set_compensate(self, s):
    #     self.compensate = s


if __name__ == '__main__':
    a = ForceSensor()
    a.render(None)

