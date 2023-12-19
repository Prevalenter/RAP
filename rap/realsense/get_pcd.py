import numpy as np

import open3d as o3d

if __name__ == '__main__':
    depth_image = np.load('depth_image.npy')
    color_image = np.load('color_image.npy')
    intri = np.load('intri_para.npy')
    width, height, fx, fy, ppx, ppy = intri
    print(depth_image.shape, color_image.shape, intri.shape)
    print(intri)

    color_raw = o3d.geometry.Image(color_image)
    depth_raw = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, ppx, ppy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.io.write_point_cloud('1.pcd', pcd)


