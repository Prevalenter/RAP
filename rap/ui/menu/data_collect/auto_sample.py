import numpy as np
import zarr
from PyQt5.QtCore import QThread
import os
import time
import pandas as pd
from pathlib import Path

from utils import zarr_io

class AutoSampleThread(QThread):
    def __init__(self, parent):
        super(AutoSampleThread, self).__init__()
        self.parent = parent

    def run(self):
        # i = 0

        # save the data
        root_dir = Path('../data/diffusion_peg_in_hole/auto')
        root_dir.mkdir(exist_ok=True)
        i = len(os.listdir(root_dir))

        is_need_prepare = True

        while self.parent.auto_sample_flag and i<200:
            # get the data
            #
            # if is_need_prepare:
            #     # to initial position
            #     self.parent.on_initial_position()
            #     time.sleep(3)
            #
            #     # to random state
            #     self.parent.on_random_position()
            #     time.sleep(3)
            #
            #     # peg contact the hole in constant force
            #     self.parent.on_run_assemble()
            #
            #     is_need_prepare = False
            #
            # if self.parent.peg_in_hole_ctrl.assemble_stage_flage==3:
            #     self.parent.on_stop_assemble()
            #
            #     print(i)
            #
            #
            #     data_i = np.zeros((200, 256, 256, 3))
            #
            #
            #     exp_dir = exp_dir.joinpath(f'{i:03}/')
            #     exp_dir.mkdir(exist_ok=True)
            #
            #     exp_path = exp_dir.joinpath('img.zarr.zip')
            #
            #     t = time.time()
            #     zarr_io.save_zarr_dict(exp_path, {"imgs": data_i})
            #     print(time.time()-t)


            # is_need_prepare = True
            # self.parent.on_clear()

            self.parent.data['imgs'] = np.random.randn(200, 256, 256, 4)*0
            self.parent.data['force_torque'] = np.random.randn(200, 6)
            self.parent.data['xyz_rot'] = np.random.randn(200, 6)


            exp_dir = root_dir.joinpath(f'{i:03}/')
            exp_dir.mkdir(exist_ok=True)

            img_path = exp_dir.joinpath('img.zarr.zip')

            t = time.time()
            zarr_io.save_zarr_dict(img_path, {"imgs": self.parent.data['imgs']})
            print(time.time()-t)

            # save ft and xyz data
            ft = np.array(self.parent.data['force_torque']).astype(np.float32)
            xyz = np.array(self.parent.data['xyz_rot']).astype(np.float32)
            # print(ft.shape, xyz.shape)
            df_ft = pd.DataFrame(ft)
            # print(ft)
            df_ft.to_csv(exp_dir.joinpath("ft.csv"), header=False, index=False)


            df_xyz = pd.DataFrame(xyz)
            df_xyz.to_csv(exp_dir.joinpath("xyz.csv"), header=False, index=False)

            self.parent.label_ctrl_auto.setText(f"Auto:    {i}")

            # time.sleep(0.5)
            i += 1



