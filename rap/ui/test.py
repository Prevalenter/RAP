import numpy as np
from scipy.spatial.transform import Rotation as R


angles = np.array([3.14, 0, 3.14])

r1 = R.from_rotvec(angles)
print(r1.as_matrix())

r2 = R.from_rotvec([0, 0, 3.14]).as_matrix() @ R.from_rotvec([3.14, 0, 0]).as_matrix()
print(r2)


