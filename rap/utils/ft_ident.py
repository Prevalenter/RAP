import numpy as np

import sys
sys.path.append('..')

from utils.trans import get_transform, to_world_axis, ft_map

def get_ft_para(path='../data/calibrate/ft.npy', static_only=False):
    data = np.load(path)

    if static_only:
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
        T_B_S_con, ft_cur = apply_force_onece(ft_cur, xyz_cur)

        A_list.append(T_B_S_con)
        F_list.append(ft_cur[:3])

    A_list = np.concatenate(A_list, axis=0)
    F_list = np.concatenate(F_list)

    rst_force = np.linalg.pinv(A_list) @ F_list

    A_list = []
    Y_list = []
    for i in range(data.shape[0]):
        F_x, F_y, F_z, T_x, T_y, T_z = data[i, 0]

        A_i = np.zeros((3, 6))
        A_i[:3, :3] = np.array([
            [0, F_z, -F_y],
            [-F_z, 0, F_x],
            [F_y, -F_x, 0]
        ])
        A_i[:3, 3:] = np.eye(3)
        Y_i = np.array([T_x, T_y, T_z]).T
        A_list.append(A_i)
        Y_list.append(Y_i)

    A_list = np.concatenate(A_list)
    Y_list = np.concatenate(Y_list)

    rst_torque = np.linalg.pinv(A_list) @ Y_list

    return rst_force, rst_torque

def apply_force_onece(ft_cur, xyz_cur):

    ft_cur = ft_map(ft_cur)

    # from sensor axis to base axis
    T_S_B = to_world_axis(xyz_cur[:3], xyz_cur[3:])
    T_B_S = np.linalg.pinv(T_S_B)
    T_B_S_con = np.concatenate([T_B_S[:3, :3], np.diag([1, 1, 1])], axis=1)

    return T_B_S_con, ft_cur

# def get_force_torque_contact_once(para, ft_cur, xyz_cur):
#     get_force_contact_once

def get_force_contact_once(para, ft_cur, xyz_cur):
    T_B_S_con, ft_cur = apply_force_onece(ft_cur, xyz_cur)
    force_estimate = T_B_S_con @ para
    force_contact = ft_cur[:3] - force_estimate
    return force_contact, force_estimate

def get_force_contact_list(para, ft_cur_list, xyz_cur_list):

    force_contact_list = []
    force_estimate_list = []
    for i in range(ft_cur_list.shape[0]):
        force_contact, force_estimate = get_force_contact_once(para, ft_cur_list[i], xyz_cur_list[i])
        force_contact_list.append(force_contact)
        force_estimate_list.append(force_estimate)
    force_contact_list = np.array(force_contact_list).reshape((-1, 3))
    force_estimate_list = np.array(force_estimate_list).reshape((-1, 3))
    return force_contact_list, force_estimate_list

def get_torque_contact_once(para, ft_cur):

    F_x, F_y, F_z, T_x, T_y, T_z = (ft_cur)
    # print(F_x, F_y, F_z, T_x, T_y, T_z )
    A_i = np.zeros((3, 6))
    A_i[:3, :3] = np.array([
        [0, F_z, -F_y],
        [-F_z, 0, F_x],
        [F_y, -F_x, 0]
    ])
    A_i[:3, 3:] = np.eye(3)

    torque_gravity = A_i @ para
    torque_contact = ft_cur[3:] - torque_gravity
    return torque_contact, torque_gravity

def get_torque_contact_list(para, ft_cur_list):
    torque_contact_list = []
    torque_estimate_list = []
    for i in range(ft_cur_list.shape[0]):
        force_contact, force_estimate = get_torque_contact_once(para, ft_cur_list[i])
        torque_contact_list.append(force_contact)
        torque_estimate_list.append(force_estimate)
    torque_contact_list = np.array(torque_contact_list).reshape((-1, 3))
    torque_estimate_list = np.array(torque_estimate_list).reshape((-1, 3))
    return torque_contact_list, torque_estimate_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = np.load('../data/calibrate/ft.npy')
    print(data.shape)

    ft_cur, xyz_cur = data[:, 0], data[:, 1]

    ft_cur_map = []
    for i in range(ft_cur.shape[0]):
        ft_cur_map.append(ft_map(ft_cur[i]))
    ft_cur_map = np.array(ft_cur_map)

    para_force, para_torque = get_ft_para()

    print('result: ', para_force.shape, para_torque.shape)
    print('gravity: ', np.linalg.norm(para_force[:3]))

    force_contact, force_estimate = get_force_contact_list(para_force, data[:, 0], data[:, 1])
    torque_contact, torque_estimate = get_torque_contact_list(para_torque, data[:, 0])
    print(force_contact.shape, force_estimate.shape, torque_contact.shape, torque_estimate.shape)

    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.subplot(3, 2, 1+2*i)
        plt.plot(ft_cur_map[:, i], c='b', label='Measurement')
        plt.plot(force_estimate[:, i], c='g', label='Prediction')
        plt.legend()
        if i==0:
            plt.title('Force')


    for i in range(3):
        plt.subplot(3, 2, 2+2*i)
        plt.plot(ft_cur[:, 3+i], c='b', label='Prediction')
        plt.plot(torque_estimate[:, i], c='g', label='Measurement')
        if i==0:
            plt.title('Torque')

    plt.subplots_adjust(hspace=0.1, right=0.95, bottom=0.2)
    plt.show()


