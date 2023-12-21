import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transform(xyz, angles):
    assert len(xyz)==3
    assert len(angles)==3
    T = np.zeros((4, 4))

    T[:3, :3] = R.from_euler("zyx", angles, False).as_matrix()

    T[:3, 3] = xyz
    T[3, 3] = 1

    return T

if __name__ == '__main__':
    T = get_transform([1, 2, 1], [12, 32, 45])
    print(T)
