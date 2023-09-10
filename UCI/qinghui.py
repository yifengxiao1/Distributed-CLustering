import numpy as np

mi = np.load('satellite/6bit_MI.npy')[:4]
pu = np.load('satellite/6bit_Purity.npy')[:4]
np.save('satellite/12bit_MI.npy', mi)
np.save('satellite/12bit_Purity.npy', pu)
print(mi[-1])