import numpy as np
import matplotlib.pyplot as plt

track='thruxton'
n=100

# view files
loaded = np.load(f'{track}/{track}_{n}.npz')
print(loaded.files)

# save image
pose_data, image = loaded['pose_data'], loaded['image']
plt.imsave(f'{track}_{n}.png', image)
