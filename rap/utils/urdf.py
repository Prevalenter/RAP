from urdfpy import URDF


import numpy
import moderngl
import math
import sys

sys.path.append('..')

from PyQt5 import QtOpenGL, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget
from ui.ui3d.arcball import ArcBallUtil
from ui.ui3d import shaders
from pyrr import Matrix44

from PyQt5 import QtWidgets

import openmesh
import numpy as np
import aspose.threed as a3d

import sys
sys.path.append('..')
from ui.ui3d.axis import Axis
from ui.ui3d.forcesensor import ForceSensor


def grid(size, steps):
    # Create grid parameters
    u = numpy.repeat(numpy.linspace(-size, size, steps), 2)
    v = numpy.tile([-size, size], steps)
    w = numpy.zeros(steps * 2)
    new_grid = numpy.concatenate([numpy.dstack([u, v, w]), numpy.dstack([v, u, w])])

    # Rotate grid
    lower_grid = 0.135
    rotation_matrix = numpy.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, lower_grid, 0]
    ])
    return numpy.dot(new_grid, rotation_matrix)


class QGLControllerWidget(QtOpenGL.QGLWidget):
    def __init__(self, up_ctrl, parent=None):
        self.parent = parent
        super(QGLControllerWidget, self).__init__(parent)

        self.up_ctrl = up_ctrl

        # Initialize OpenGL parameters
        self.bg_color = (0.1, 0.1, 0.1, 0.1)
        self.color_alpha = 1.0
        self.new_color = (1.0, 1.0, 1.0, self.color_alpha)
        self.fov = 60.0
        self.camera_zoom = 2.0
        self.setMouseTracking(True)
        self.wheelEvent = self.update_zoom
        self.is_wireframe = False
        self.texture = None
        self.cell = 50
        self.size = 20
        self.grid = grid(self.size, self.cell)
        self.grid_alpha_value = 1.0

    def initializeGL(self):
        # Create a new OpenGL context
        self.ctx = moderngl.create_context()

        # Create the shader program
        self.prog = self.ctx.program(
            vertex_shader=shaders.vertex_shader,
            fragment_shader=shaders.fragment_shader
        )

        self.set_scene()
        # self.update_grid()
        # self.set_mesh(None)

    def set_scene(self):

        print(list(self.prog))

        # Setting shader parameters
        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.prog["Texture"].value = 0
        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 1.0)

        # Setting mesh parameters
        self.mesh_test = 0
        self.mesh = None
        self.vbo = self.ctx.buffer(self.grid.astype('f4'))
        self.vao2 = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_position')

        # Setting ArcBall parameters
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        self.center = numpy.zeros(3)
        # self.center = np.ones(3)
        self.scale = 1.0

        self.rbt_urdf = URDF.load("../data/estun/estun.urdf")
        self.set_rbt()
        #
        self.set_rbt([0.5]*6)


    def set_rbt(self, angles=None):
        # rbt_angles = np.array([0, 0, 0, 0, 0, 0])
        rbt_angles_dict = {}
        for idx, j in enumerate(self.rbt_urdf.actuated_joints):
            if angles is None:
                rbt_angles_dict[j.name] = 0
            else:
                rbt_angles_dict[j.name] = angles[idx]
            # print(j.name, j.joint_type, dir(j.limit), j.limit.lower, j.limit.upper, j.mimic)
        #
        fk = self.rbt_urdf.visual_trimesh_fk(rbt_angles_dict)
        key_list = list(fk.keys())
        self.rbt_vao_list = []
        for key in key_list:
            tri_mesh = key
            pos = fk[key]
            print('-'*50)
            # vertex_normals vertices face_adjacency faces
            print(tri_mesh.vertex_normals.shape, tri_mesh.faces.shape, tri_mesh.vertices.shape, pos.shape)

            # apply pos
            mesh_vert = tri_mesh.vertices
            mesh_vert_normal = tri_mesh.vertex_normals

            print(pos)
            mesh_vert = mesh_vert @ pos[:3, :3].T + pos[:3, 3]
            mesh_vert_normal = mesh_vert_normal @ pos[:3, :3].T + pos[:3, 3]

            index_buffer = self.ctx.buffer(numpy.array(tri_mesh.faces, dtype="u4").tobytes())
            vao_content = [(self.ctx.buffer(numpy.array(mesh_vert, dtype="f4").tobytes()), '3f', 'in_position'),
                           (self.ctx.buffer(numpy.array(mesh_vert_normal, dtype="f4").tobytes()), '3f', 'in_normal')]
            rbt_vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)
            self.rbt_vao_list.append(rbt_vao)

    def paintGL(self):
        # OpenGL loop
        self.ctx.clear(*self.bg_color)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.wireframe = self.is_wireframe
        # if self.mesh is None:
        #     return

        # Update projection matrix loop
        self.aspect_ratio = self.width() / max(1.0, self.height())
        proj = Matrix44.perspective_projection(self.fov, self.aspect_ratio, 0.1, 1000.0)

        lookat = Matrix44.look_at(
            (self.camera_zoom, 0.0, 0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        self.arc_ball.Transform[3, :3] = -self.arc_ball.Transform[:3, :3].T @ self.center
        self.mvp.write((proj * lookat * self.arc_ball.Transform).astype('f4'))

        # if self.mesh is not None:
        #     # Render mesh loop
        #     self.color.value = (0.9, 0.9, 0.9, 0.9)
        #     self.vao.render()
        self.color.value = (0.9, 0.9, 0.9, 0.9)
        # self.vao.render()
        for rbt_vao in self.rbt_vao_list:

            rbt_vao.render()

        # Render grid loop

        self.color.value = self.new_color
        self.vertices_reigon = np.array([
            [1, 1, 1], [1, 1, -1],
            [1, 1, -1], [1, -1, -1],
            [1, -1, -1], [1, -1, 1],
            [1, -1, 1], [1, 1, 1],

            [-1, 1, 1], [-1, 1, -1],
            [-1, 1, -1], [-1, -1, -1],
            [-1, -1, -1], [-1, -1, 1],
            [-1, -1, 1], [-1, 1, 1],

            [1, 1, 1], [-1, 1, 1],
            [-1, 1, 1], [-1, -1, 1],
            [-1, -1, 1], [1, -1, 1],
            [1, -1, 1], [1, 1, 1],

            [1, 1, -1], [-1, 1, -1],
            [-1, 1, -1], [-1, -1, -1],
            [-1, -1, -1], [1, -1, -1],
            [1, -1, -1], [1, 1, -1],
        ], dtype=np.float32)
        self.vbo_reigon = self.ctx.buffer(self.vertices_reigon)
        self.vao_reigon = self.ctx.simple_vertex_array(self.prog, self.vbo_reigon, 'in_position')
        # Render reigon loop
        self.color.value = (1.0, 1.0, 1.0, self.grid_alpha_value)
        self.vao_reigon.render(moderngl.LINES)

        self.up_ctrl.axis.render(self)
        self.up_ctrl.axis_ori.render(self)
        self.up_ctrl.ft.render(self)

    def set_mesh(self, new_mesh):
        if new_mesh is None:
            self.set_scene()
            return
        self.mesh = new_mesh
        self.mesh.update_normals()

        # Creates an index buffer
        index_buffer = self.ctx.buffer(numpy.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())

        # Creates a list of vertex buffer objects (VBOs)
        vao_content = [(self.ctx.buffer(numpy.array(self.mesh.points()/300, dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(numpy.array(self.mesh.vertex_normals(), dtype="f4").tobytes()), '3f', 'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)
        # self.init_arcball()

    def init_arcball(self):
        # Create ArcBall
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        mesh_points = self.mesh.points()
        bounding_box_min = numpy.min(mesh_points, axis=0)
        bounding_box_max = numpy.max(mesh_points, axis=0)
        self.center = 0.5*(bounding_box_max+bounding_box_min)
        self.scale = numpy.linalg.norm(bounding_box_max-self.center)
        self.arc_ball.Transform[:3, :3] /= self.scale
        self.arc_ball.Transform[3, :3] = -self.center/self.scale

    # -------------- GUI interface --------------
    def change_light_color(self, color, alpha=1):
        color = color + (alpha,)
        print(color)
        self.color.value = color
        self.new_color = color

    def update_alpha(self, alpha):
        self.color_alpha = (alpha*0.01)
        color_list = list(self.new_color)
        color_list[-1] = self.color_alpha
        self.new_color = tuple(color_list)

    def update_grid_alpha(self, alpha):
        self.grid_alpha_value = (alpha*0.01)

    def background_color(self, color):
        self.bg_color = color

    def update_fov(self, num):
        self.fov = num
        self.camera_zoom = self.camera_distance(num)
        self.update()

    @staticmethod
    def camera_distance(num):
        return 1 / (math.tan(math.radians(num / 2)))

    def update_grid_cell(self, cells):
        self.cell = cells
        self.grid = grid(self.size, self.cell)
        self.update_grid()

    def update_grid_size(self, size):
        self.size = size
        self.grid = grid(self.size, self.cell)
        self.update_grid()

    def update_grid(self):
        self.vbo = self.ctx.buffer(self.grid.astype('f4'))
        self.vao2 = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_position')

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)
        self.arc_ball.setBounds(width, height)
        return

    def make_wireframe(self):
        self.is_wireframe = True

    def make_solid(self):
        self.is_wireframe = False

    # Input handling
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftDown(event.x(), event.y())
        elif event.buttons() & QtCore.Qt.RightButton:
            self.prev_x = event.x()
            self.prev_y = event.y()

    def mouseReleaseEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftUp()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onDrag(event.x(), event.y())

    def update_zoom(self, event):
        self.camera_zoom += event.angleDelta().y() * 0.001
        if self.camera_zoom < 0.1:
            self.camera_zoom = 0.1
        self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onDrag(event.x(), event.y())
        elif event.buttons() & QtCore.Qt.RightButton:
            x_movement = event.x() - self.prev_x
            y_movement = event.y() - self.prev_y
            self.center[0] -= x_movement * 0.01
            self.center[1] += y_movement * 0.01
            self.update()
            self.prev_x = event.x()
            self.prev_y = event.y()

    def setRotX(self, val):
        self.axis.angles[0] = val/100

    def setRotY(self, val):
        self.axis.angles[1] = val / 100

    def setRotZ(self, val):
        self.axis.angles[2] = val / 100

def load_stl(path):
    a3d_obj = a3d.Scene.from_file(path)
    a3d_obj.save('tmp.obj')

    return openmesh.read_trimesh('tmp.obj')

class Ui3dWindow(QtWidgets.QMainWindow):

    def __init__(self, up_ctrl):
        QtWidgets.QMainWindow.__init__(self)  # call the init for the parent class

        self.resize(600, 600)
        self.setMinimumWidth(800)
        self.setWindowTitle('Hello OpenGL App')

        self.up_ctrl = up_ctrl

        self.glWidget = QGLControllerWidget(up_ctrl=up_ctrl)

        self.initGUI()

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()

        # self.glWidget.set_mesh(None)

    def update(self):
        pass

    def sim_sent(self):
        print('sim sent')

    def keyPressEvent(self, a0) -> None:
        self.glWidget.ctrl_on = True
        print('key on: ', self.glWidget.ctrl_on)

        file_name = 'link1.obj'
        # stl_fn = "/home/lx/Documents/lx/work/hust/data/estun/meshes/link2.STL"
        stl_fn = "/home/lx/Documents/lx/work/hust/data/coordinate/HandFrame3.stl"
        mesh = load_stl(stl_fn)

        # mesh = openmesh.read_trimesh(file_name)
        self.glWidget.set_mesh(mesh)

    def keyReleaseEvent(self, a0):
        self.glWidget.ctrl_on = False
        print('key off: ', self.glWidget.ctrl_on)

    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        gui_layout.addWidget(self.glWidget)


class PlotMainWindow(QWidget):
    def __init__(self, up_ctrl=None):
        super().__init__()
        self.xyz_tgt = [0.5, 0.3, 0, 0, 0, 0]
        self.xyz_cur = [0.45, 0.1, 0, 0, 0, 0]
        # self.ft_cur = [0]*6

        self.axis = Axis(scale=0.1)
        self.axis_ori = Axis()
        self.ft = ForceSensor()

        self.plot = Ui3dWindow(up_ctrl=self)
        self.plot.setParent(self)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        self.plot.update()

        # self.ft_cur


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    win = PlotMainWindow()
    win.show()

    sys.exit(app.exec_())





