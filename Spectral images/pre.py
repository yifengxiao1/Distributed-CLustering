import numpy as np
from scipy.io import loadmat

gt = loadmat('data set/PaviaU_gt.mat')['paviaU_gt'].flatten()
input_image = loadmat('data set/PaviaU.mat')['paviaU'].reshape(-1, 103)
zero_indexes = np.where(gt==0)
new_gt = np.delete(gt, zero_indexes)
new_image = np.delete(input_image, zero_indexes, axis=0)
print(np.unique(new_gt))
# for i in range(len(new_gt)-200):
#     temp = new_gt[i: i+200]
#     temp = np.unique(temp)
#     print(temp)
