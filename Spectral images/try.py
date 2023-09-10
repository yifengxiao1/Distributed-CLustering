from scipy.io import loadmat
import numpy as np
import spectral as spy
from sklearn.metrics import adjusted_mutual_info_score as MI
import torch
#
# Purity = np.load('Results/paviaC/1/8bit_Purity.npy', allow_pickle=True)
# MI = np.load('Results/paviaC/1/8bit_MI.npy', allow_pickle=True)
#
# index = np.argmax(MI)
# print(index)
# print(MI[index])
# print(Purity[index])


a = loadmat('data set/Salinas_corrected.mat')['salinas_corrected'].reshape(-1, 204).astype(np.float16)
print(a)

