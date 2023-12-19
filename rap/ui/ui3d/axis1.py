import numpy as np
from scipy.spatial.transform import Rotation as R
import moderngl

class Axis1:
    def __init__(self, seq='zyx', scale=1):
        self.xyz = np.zeros(3)
        self.angles = np.zeros(3)
        self.scale = scale
        self.seq = seq

    def render(self, up_ctrl):
        rst = self.calucu()

        up_ctrl.color.value = up_ctrl.new_color
        for i in range(3):
            up_ctrl.vao_axis_simple = []
            up_ctrl.vbo_axis_simple = []
            tmp = np.zeros(3)
            tmp[i] = 1
            up_ctrl.vbo_axis_simple.append(up_ctrl.ctx.buffer(np.array([
                                    self.xyz, self.xyz+rst[:3, i]*self.scale ], dtype=np.float32)))
            up_ctrl.vao_axis_simple.append( up_ctrl.ctx.simple_vertex_array(up_ctrl.prog, up_ctrl.vbo_axis_simple[-1], 'in_position') )
            # Render reigon loop
            tmp[tmp==0] = 0.2
            up_ctrl.color.value = (*tmp, up_ctrl.grid_alpha_value)
            up_ctrl.vao_axis_simple[-1].render(moderngl.LINES)


    def calucu(self):
        angles = self.angles.copy()
        angles[0] = -angles[0]
        angles[2] = -angles[2]
        return R.from_euler(self.seq, angles, False).as_matrix()

    def set_axis(self, xyz, angles):
        if xyz is not None:
            self.xyz = xyz
        if angles is not None:
            self.angles = angles

    def set_sacle(self, scale):
        self.scale = scale

if __name__ == '__main__':
    a = Axis1()
    a.calucu()

    a.set_axis(None, np.array([0, 0, 1])*(np.pi/4))
    a.render()
