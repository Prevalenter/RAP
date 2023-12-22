import numpy as np

import sys
sys.path.append('..')

from utils.trans import get_transform, to_world_axis, ft_map

def get_ft_para(path='../data/calibrate/ft.npy'):
    data = np.load(path)

    # get the static data
    xyz_cur = data[:, 1, 3:]
    xyz_cur = xyz_cur[1:] - xyz_cur[:-1]
    xyz_cur_norm = np.linalg.norm(xyz_cur, axis=1)

    select_mask = (xyz_cur_norm<0.005) * (xyz_cur_norm>0)
    data = data[1:][select_mask]

    print('using data shape is: ', data.shape)
    A_list = []
    F_list = []
    for i in range(data.shape[0]):
        ft_cur, xyz_cur = data[i]
        T_B_S_con, ft_cur = apply_onece(ft_cur, xyz_cur)

        A_list.append(T_B_S_con)
        F_list.append(ft_cur[:3])

    A_list = np.concatenate(A_list, axis=0)
    F_list = np.concatenate(F_list)

    rst = np.linalg.pinv(A_list) @ F_list
    return rst

def apply_onece(ft_cur, xyz_cur):

    ft_cur = ft_map(ft_cur)

    # from sensor axis to base axis
    T_S_B = to_world_axis(xyz_cur[:3], xyz_cur[3:])
    T_B_S = np.linalg.pinv(T_S_B)
    T_B_S_con = np.concatenate([T_B_S[:3, :3], np.diag([1, 1, 1])], axis=1)

    return T_B_S_con, ft_cur

def get_force_conttace_once(para, ft_cur, xyz_cur):
    T_B_S_con, ft_cur = apply_onece(ft_cur, xyz_cur)
    force_contact = ft_cur[:3] - T_B_S_con @ para
    return force_contact

def get_force_contact_list(para, ft_cur_list, xyz_cur_list):

    force_contact_list = []
    for i in range(ft_cur_list.shape[0]):
        force_contact_list.append(get_force_conttace_once(para, ft_cur_list[i], xyz_cur_list[i]))

    return np.array(force_contact_list)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = np.load('../data/calibrate/ft.npy')
    print(data.shape)

    para = get_ft_para()

    print('result: ', para)
    print('gravity: ', np.linalg.norm(para[:3]))

    error = get_force_contact_list(para, data[:, 0], data[:, 1]).reshape((-1, 3))
    print(error.shape)


    # ft_cur, xyz_cur = data[0, 0], data[0, 1]
    # T_B_S_con, ft_cur = apply_onece(ft_cur, xyz_cur)
    # force_contact = ft_cur[:]
    # print(T_B_S_con.shape, ft_cur.shape, xyz_cur.shape)
    plt.plot(np.linalg.norm(error, axis=1))
    plt.show()
