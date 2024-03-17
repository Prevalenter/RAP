import matplotlib.pyplot as plt
import numpy as np

data = np.load('reward_list_TD3_Gear1-v0_0.npy')

print(data)
print((data!=0).sum()/len(data))

plt.plot(data)
plt.show()

