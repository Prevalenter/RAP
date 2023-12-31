import pyrealsense2 as rs
import numpy as np
import cv2


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

    def run_frame(self):
        # try:
            depth_image = []
            color_image = []
            for idx in range(len(get_dev_id_list())):
                depth_image_i, color_image_i = self.get_rgbd(self.dev_list[idx])
                depth_image.append(depth_image_i)
                color_image.append(color_image_i)

            depth_image = np.vstack(depth_image)
            color_image = np.vstack(color_image)

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            # clipping_distance = 1500
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            # plot the image
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()

    def clean(self):
        for p in self.dev_list:
            p.stop()

if __name__ == '__main__':

    multi_cam = MultiReansense()
    for i in range(100):
        multi_cam.run_frame()
