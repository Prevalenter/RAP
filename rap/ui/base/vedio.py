import pyrealsense2 as rs
import numpy as np
# import cv2

import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPixmap
import sys


dev_id_list = ['147122075207',
               '147322070524']

def get_dev_id_list():
    dev_id_list = []
    ctx = rs.context()
    for idx, dev in enumerate(ctx.query_devices()):
        sensor_sn = dev.get_info(rs.camera_info.serial_number)
        dev_id_list.append(sensor_sn)
    return dev_id_list

class MultiReansense:
    def __init__(self):
        dev_id_detected = get_dev_id_list()
        print(dev_id_detected)
        assert len(dev_id_detected)==len(dev_id_list)
        self.cam_enable()

    def cam_enable(self):
        dev_list = []
        intr_list = []
        for idx in range(len(get_dev_id_list())):
            print('enabling: ', dev_id_list[idx])
            pipeline = rs.pipeline()
            dev_list.append(pipeline)

            config = rs.config()
            # config.enable_device(dev_id)
            config.enable_device(dev_id_list[idx])
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            profile = pipeline_profile.get_stream(rs.stream.depth)
            intr = profile.as_video_stream_profile().get_intrinsics()
            intr_list.append(intr)

            profile = pipeline.start(config)
        self.dev_list, self.intr_list = dev_list, intr_list

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_rgbd(self, pipeline):
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def get_frame(self):
        # try:
        depth_image = []
        color_image = []
        for idx in range(len(get_dev_id_list())):
            depth_image_i, color_image_i = self.get_rgbd(self.dev_list[idx])
            depth_image.append(depth_image_i)
            color_image.append(color_image_i)

        depth_image = np.vstack(depth_image)
        color_image = np.vstack(color_image)


        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        return images

#             # plot the image
#             cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#             cv2.imshow('Align Example', images)
#             key = cv2.waitKey(1)
#             # Press esc or 'q' to close the image window
#             if key & 0xFF == ord('q') or key == 27:
#                 cv2.destroyAllWindows()
#
#     def clean(self):
#         for p in self.dev_list:
#             p.stop()
#


class CamThread(QThread):
    changePixmap = pyqtSignal(QImage)
    multi_cam = MultiReansense()

    def run(self):
        while True:
            print('thread running')
            # img_data = np.zeros((800, 800, 3))
            img = multi_cam.get_frame()
            w, h, c = img.shape
            # bytesPerLine = 800 * 800 * 3
            convertToQtFormat = QImage(img_data, w, h, QImage.Format_RGB888)
            # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(convertToQtFormat)
            self.changePixmap.emit(convertToQtFormat)
            time.sleep(0.5)
            # try:
            #     img_data = np.zeros((800, 800, 3))
            #     bytesPerLine = 800*800*3
            #     convertToQtFormat = QImage(img_data, 800, 800, bytesPerLine, QImage.Format_RGB888)
            #     p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            #     # self.changePixmap.emit(p)
            # except Exception as e:
            #     raise e

        # cap = cv2.VideoCapture(0)
        # while True:
        #     ret, frame = cap.read()
        #     if ret:
        #         # https://stackoverflow.com/a/55468544/6622587
        #         rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         h, w, ch = rgbImage.shape
        #         bytesPerLine = ch * w
        #         convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        #         p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        #         self.changePixmap.emit(p)




class VedioWidget(QDialog):
    def __init__(self, up_ctrl=None):
        super().__init__()
        self.up_ctrl = up_ctrl

        self.step_index = 0
        self.ui_init()

    def test_data(self):
        t = self.para['dt'] * self.step_index
        return [np.sin(t+i) for i in range(6)]

    def ui_init(self):
        state_box = QGroupBox("Dataset sample")
        state_box.setMinimumWidth(640)
        layout_state = QGridLayout()
        self.angle_le_list = []

        layout_state.addWidget(QLabel("Camera: "), 0, 0)
        # layout_state.addWidget(QLabel("Ylim Low"), 1, 0)

        # self.btn_set = QPushButton('Set')
        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        layout_state.addWidget(self.btn_start, 0, 1)
        layout_state.addWidget(self.btn_stop, 0, 2)


        self.label = QLabel(self)
        self.label.setMaximumWidth(640)
        self.label.setMinimumHeight(480)

        state_box.setLayout(layout_state)

        container = QVBoxLayout()
        container.addWidget(state_box)
        container.addWidget(self.label)
        container.addStretch(2)
        self.setLayout(container)

        th = CamThread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))



class MyWindow123(QWidget):
    def __init__(self):
        super().__init__()
        self.xyz_cur = [0] * 6
        self.angle_cur = [0] * 6
        self.ft_cur = [0] * 6

        self.vedio_widget = VedioWidget(up_ctrl=self)
        self.vedio_widget.setParent(self)

        self.resize(800, 680)

    def update(self, tgt=False, cur=False, sent_tgt=False):
        if tgt:
            self.vedio_widget.update_tgt()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow123()
    # w = ConnectWidget()
    w.show()
    app.exec_()




# if __name__ == '__main__':
#
#     multi_cam = MultiReansense()
#     for i in range(100):
#         multi_cam.run_frame()
