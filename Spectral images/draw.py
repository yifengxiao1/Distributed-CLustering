import numpy as np
from matplotlib import pyplot as plt

def stabilize(array):
    new_array = np.zeros(int(len(array)/10))
    for i in range(int(len(array)/10)):
        new_array[i] = np.mean(array[10*i:10*i+10])
    return new_array


Purity = np.load('Results/Salinas/24bit_Purity.npy', allow_pickle=True)
MI = np.load('Results/Salinas/24bit_MI.npy', allow_pickle=True)
plt.plot(Purity, label='Purity of hash codes', color='g')
plt.plot(MI, label='MI of hash codes', color='b')
# plt.plot(stabilize(bs_Accuracy), label='Accuracy of raw data', color='b')
plt.xlabel('Iterations')
plt.ylabel("1")
plt.legend()
plt.show()
print(np.max(Purity))
print(np.max(MI))
