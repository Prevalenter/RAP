import numpy as np
from scipy.spatial.transform import Rotation as R
import moderngl

import sys
sys.path.append("../..")
from utils.trans import get_transform

class Axis:
    def __init__(self, seq='zyx', scale=1):
        self.xyz = np.zeros(3)
        self.angles = np.zeros(3)
        self.scale = scale
        self.seq = seq

        self.set_sacle(scale)

        # print(self.axis_base)

    def render(self, gl_ctrl):
        # rst =
        T = self.calucu()
        end = T @ self.axis_base
        end = end[:3, :3]

        if gl_ctrl is not None:
            gl_ctrl.color.value = gl_ctrl.new_color
            for i in range(3):
                gl_ctrl.vao_axis_simple = []
                gl_ctrl.vbo_axis_simple = []
                tmp = np.zeros(3)
                tmp[i] = 1
                # gl_ctrl.vbo_axis_simple.append(gl_ctrl.ctx.buffer(np.array([
                #                         self.xyz, self.xyz+rst[:3, i]*self.scale ], dtype=np.float32)))
                gl_ctrl.vbo_axis_simple.append(gl_ctrl.ctx.buffer(np.array([
                                        self.xyz, end[:, i] ], dtype=np.float32)))

                gl_ctrl.vao_axis_simple.append( gl_ctrl.ctx.simple_vertex_array(gl_ctrl.prog, gl_ctrl.vbo_axis_simple[-1], 'in_position') )
                # Render reigon loop
                tmp[tmp==0] = 0.2
                gl_ctrl.color.value = (*tmp, gl_ctrl.grid_alpha_value)
                gl_ctrl.vao_axis_simple[-1].render(moderngl.LINES)


    def calucu(self):
        angles = self.angles.copy()
        angles[0] = -angles[0]
        angles[2] = -angles[2]

        return get_transform(self.xyz, angles)



    def set_axis(self, xyz=None, angles=None):
        if xyz is not None:
            self.xyz = xyz
        if angles is not None:
            self.angles = angles

    def set_sacle(self, scale):
        self.scale = scale

        self.axis_base = np.zeros((4, 3))
        self.axis_base[3] = 1
        self.axis_base[:3, :3] = np.diag([1, 1, 1]) * scale

if __name__ == '__main__':
    a = Axis()
    a.calucu()

    a.set_axis(None, np.array([0, 0, 1])*(np.pi/4))
    a.render(None)
