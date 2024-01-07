import pyrealsense2 as rs
import numpy as np
import cv2

def get_dev_id_list():
    dev_id_list = []
    ctx = rs.context()
    for idx, dev in enumerate(ctx.query_devices()):
        sensor_sn = dev.get_info(rs.camera_info.serial_number)
        dev_id_list.append(sensor_sn)
    return dev_id_list


class SingleReansense:
    def __init__(self, dev_id):
        self.dev_id_detected = get_dev_id_list()
        self.dev_id = dev_id
        print(dev_id, self.dev_id_detected)

        assert self.dev_id in self.dev_id_detected
        self.cam_enable()

    def cam_enable(self):
        pipeline = rs.pipeline()

        config = rs.config()
        # config.enable_device(dev_id)
        config.enable_device(self.dev_id)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        profile = pipeline_profile.get_stream(rs.stream.depth)
        intr = profile.as_video_stream_profile().get_intrinsics()

        profile = pipeline.start(config)

        self.pipeline, self.intr = pipeline, intr

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_rgbd(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

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

        depth_image = []
        color_image = []

        depth_image_i, color_image_i = self.get_rgbd()
        depth_image.append(depth_image_i)
        color_image.append(color_image_i)

        depth_image = np.vstack(depth_image)
        color_image = np.vstack(color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))

        return images