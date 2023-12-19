import open3d as o3d

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("1.pcd")

    print(pcd)

    o3d.visualization.draw_geometries([pcd])

