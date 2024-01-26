import numpy as np
import sys
import matplotlib.pyplot as plt
from skvideo.io import vwrite
sys.path.append('../../..')

from utils import zarr_io

if __name__ == '__main__':

    data = zarr_io.load_zarr_dict('../../../data/diffusion_peg_in_hole/auto/000/data.zarr.zip')
    assemble_stage = data['assemble_stage']
    mask = assemble_stage==2
    print(data.keys())
    imgs = data['img']
    force_torque = data['force_torque']
    xyz_rot_real = data['xyz_rot_real']
    xyz_rot_tgt = data['xyz_rot_tgt']
    print(imgs.shape, assemble_stage.shape, force_torque.shape,
          xyz_rot_real.shape, xyz_rot_tgt.shape)

    for i in range(6):
        plt.subplot(6, 1, 1+i)
        plt.plot(xyz_rot_real[:, i], label='real')
        plt.plot(xyz_rot_tgt[:, i], label='tgt')
    plt.legend()
    plt.show()



    breakpoint()


    imgs_show = np.concatenate([imgs[:, 0, :, :, :3], imgs[:, 1, :, :, :3]], axis=2)
    print(imgs_show.shape)
    # plt.imshow(imgs_show[0])
    # plt.show()
    # vwrite('vis.mp4', imgs_show.astype(np.uint8))

    imgs_show = []
    # imgs.shape[0]
    for idx in range(imgs.shape[0]):
        print(idx)

        fig = plt.figure(figsize=(8, 8))
        canvas = plt.gca().figure.canvas


        for i in range(2):
            plt.subplot(2, 2, 1+i)
            plt.imshow(imgs[idx, i, :, :, :3])


        for i in range(6):
            plt.subplot(12, 2, 12+1+i*2)
            plt.plot(xyz_rot_real[:, i], c='k')
            plt.plot([idx, idx], [xyz_rot_real[:, i].min(), xyz_rot_real[:, i].max()], c='r')
            if i==0:
                plt.title('XYZ-Rot')

        for i in range(6):
            plt.subplot(12, 2, 12+2+i*2)
            plt.plot(force_torque[:, i], c='k')
            plt.plot([idx, idx], [force_torque[:, i].min(), force_torque[:, i].max()], c='r')
            if i==0:
                plt.title('Force-Tensor')
        canvas.draw()

        frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        print(frame.shape)
        plt.close()

        imgs_show.append(frame)
    imgs_show = np.array(imgs_show).astype(np.uint8)
    vwrite('vis.mp4', imgs_show)

    # plt.plot(assemble_stage)
    # plt.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(imgs[0, 0])
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(imgs[-1, 0])
    #
    # plt.show()
