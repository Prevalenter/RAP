import numpy as np
import zarr
from PyQt5.QtCore import QThread
import os
import time
import pandas as pd
from pathlib import Path
from PyQt5.QtCore import *
from utils import zarr_io

class AutoSampleThread(QThread):
    start_run = pyqtSignal()
    stop_run = pyqtSignal()
    start_recoder = pyqtSignal()
    stop_recoder = pyqtSignal()

    def __init__(self, parent):
        super(AutoSampleThread, self).__init__()
        self.parent = parent

    def run(self):
        print('AutoSampleThread run')

        # save the data
        root_dir = Path('../data/diffusion_peg_in_hole/auto')
        root_dir.mkdir(exist_ok=True)
        i = len(os.listdir(root_dir))

        is_need_prepare = True

        while self.parent.auto_sample_flag and i<200:
            # get the data
            if is_need_prepare:
                # to initial position
                self.parent.on_initial_position()
                self.parent.label_ctrl_auto.setText(f"Auto: initial    {i}")
                time.sleep(3)

                # to random state
                self.parent.on_random_position()
                self.parent.label_ctrl_auto.setText(f"Auto: random    {i}")
                time.sleep(3)

                # peg contact the hole in constant force
                # self.parent.on_run_assemble()
                self.parent.peg_in_hole_ctrl.assemble_stage_flage = 0
                self.start_run.emit()
                self.start_recoder.emit()
                # time.sleep(3)

                is_need_prepare = False
                self.parent.label_ctrl_auto.setText(f"Auto: assem    {i}")

            if self.parent.peg_in_hole_ctrl.assemble_stage_flage==3 and not is_need_prepare:
                # self.parent.on_stop_assemble()
                self.stop_run.emit()
                self.stop_recoder.emit()
                print(i)
                self.parent.label_ctrl_auto.setText(f"Auto: saving    {i}")

                # data_i = np.zeros((200, 256, 256, 3))

                exp_dir = root_dir.joinpath(f'{i:03}/')
                exp_dir.mkdir(exist_ok=True)

                exp_path = exp_dir.joinpath('data.zarr.zip')

                t = time.time()
                # zarr_io.save_zarr_dict(exp_path, {"imgs": data_i}
                # assemble_stage = self.parent.data['assemble_stage']

                data_saved = {}
                mask = np.array(self.parent.data['assemble_stage'])==2

                # min_len = 10000000
                # for k in self.parent.data:
                #     min_len = min(min_len, self.parent.data[k].shape[0])
                min_len = min([len(self.parent.data[k]) for k in self.parent.data if k!='idx'])

                for k in self.parent.data:
                    if k=='idx': continue
                    print(f'before {k}', np.array(self.parent.data[k]).shape)
                    data_saved[k] = np.array(self.parent.data[k][:min_len])[mask[:min_len]]
                    print(data_saved[k].shape)

                # zarr_io.save_zarr_dict(exp_path, {"imgs": np.array(self.parent.data['img']) })
                zarr_io.save_zarr_dict( exp_path, data_saved )

                # save ft and xyz data
                ft = np.array(self.parent.data['force_torque']).astype(np.float32)
                xyz = np.array(self.parent.data['xyz_rot_real']).astype(np.float32)
                df_ft = pd.DataFrame(ft)
                df_ft.to_csv(exp_dir.joinpath("ft.csv"), header=False, index=False)

                df_xyz = pd.DataFrame(xyz)
                df_xyz.to_csv(exp_dir.joinpath("xyz.csv"), header=False, index=False)

                print(time.time()-t)

                is_need_prepare = True
                self.parent.on_clear()

                # img_path = exp_dir.joinpath('img.zarr.zip')
                #
                # t = time.time()
                # imgs = np.array(self.parent.data['img'])
                # print('imgs', imgs.shape)
                # zarr_io.save_zarr_dict(img_path, {"imgs": imgs})
                # print(time.time()-t)





                # time.sleep(0.5)
                i += 1

            time.sleep(0.1)



