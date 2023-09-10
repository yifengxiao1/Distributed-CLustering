import numpy as np
from matplotlib import pyplot as plt

# def stabilize(array):
#     new_array = np.zeros(int(len(array)/10))
#     for i in range(int(len(array)/10)):
#         new_array[i] = np.mean(array[10*i:10*i+10])
#     return new_array


Purity = np.load('实验结果/S1/64bit_Purity.npy', allow_pickle=True)[-5:]
MI = np.load('实验结果/S1/64bit_MI.npy', allow_pickle=True)[-5:]
# plt.plot(Purity, label='Purity of hash codes', color='g')
# plt.plot(MI, label='MI of hash codes', color='b')
# # plt.plot(stabilize(bs_Accuracy), label='Accuracy of raw data', color='b')
# plt.xlabel('Iterations')
# plt.ylabel('Performance')
# plt.legend()
# plt.show()
print(np.mean(Purity))
print(np.mean(MI))

# def RER(Loss):
#     max = np.max(Loss)
#     min = np.min(Loss)
#     Loss = (Loss - min) / (max - min)
#     return list(Loss)
#
# Loss100 = np.load('实验结果/S5/64bit_Loss_100.npy', allow_pickle=True)
# Loss100 = RER(Loss100)
# # Loss100_30 = np.load('实验结果/S3/64bit_Loss_100_30.npy', allow_pickle=True)
# # Loss100_30 = RER(Loss100_30)
# Loss80 = np.load('实验结果/S5/64bit_Loss_80.npy', allow_pickle=True)
# Loss80 = RER(Loss80)
# Loss60 = np.load('实验结果/S5/64bit_Loss_60.npy', allow_pickle=True)
# Loss60 = RER(Loss60)
# Loss40 = np.load('实验结果/S5/64bit_Loss_40.npy', allow_pickle=True)
# Loss40 = RER(Loss40)
# Loss20 = np.load('实验结果/S5/64bit_Loss_20.npy', allow_pickle=True)
# Loss20 = RER(Loss20)
#
# plt.rcParams['figure.figsize'] = (12.0, 8.0)
# plt.plot(range(1, 51), Loss100, label='s=100', color='purple')
# plt.scatter(range(1, 51), Loss100, color='purple')
# # plt.plot(range(1, 31), Loss100_30, label='s=100', color='r')
# # plt.scatter(range(1, 31), Loss100_30, color='r')
# plt.plot(range(1, 51),Loss80, label='s=80', color='b')
# plt.scatter(range(1, 51), Loss80, color='b')
# plt.plot(range(1, 51),Loss60, label='s=60', color='g')
# plt.scatter(range(1, 51), Loss60, color='g')
# plt.plot(range(1, 51),Loss40, label='s=40', color='y')
# plt.scatter(range(1, 51), Loss40, color='y')
# plt.plot(range(1, 51),Loss20, label='s=20', color='r')
# plt.scatter(range(1, 51), Loss20, color='r')
#
#
# # plt.plot(stabilize(bs_Accuracy), label='Accuracy of raw data', color='b')
# plt.xlabel('Iterations',fontsize = 40)
# plt.ylabel('RER',fontsize = 40)
# plt.legend(fontsize=19)
# plt.xlim(0,50)
# plt.ylim(0,1)
# plt.yticks([0,0.5,1],fontsize = 20)
# plt.xticks([0,10,20,30,40,50],fontsize = 20)
# plt.savefig("图/S5.png", dpi=600,format="png")
# plt.show()

