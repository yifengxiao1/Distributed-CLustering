import matplotlib as plt
import numpy as np


# def generate_a_cluster(m, d):
#     T = np.random.normal(size=(m, d))
#     T /= np.linalg.norm(T, ord='fro')
#     u = np.random.uniform(low=(-20)/np.log(m), high=20/np.log(m), size=m)
#     cluster = np.zeros((2500, m))
#     for i in range(2500):
#         z = np.random.normal(size=d)
#         e = np.random.normal(loc=0, scale=1 / (10 * m), size=m)
#         cluster[i, :] = np.dot(T, z) + u + e
#     return cluster
#
#
# for i in range(9):
#     for j in range(20):
#         if i == 0 and j == 0:
#             data = generate_a_cluster(512, 2)
#             label = np.zeros(2500) + i * 20 + j
#         else:
#             data = np.concatenate((data, generate_a_cluster(512, 2 ** (i+1))), axis=0)
#             label = np.concatenate((label, np.zeros(2500) + i * 20 + j), axis=0)
#
# order = np.random.permutation(len(label))
# data = data[order]
# label = label[order]
# np.save('测试数据/G9data.npy', data)
# np.save('测试数据/G9label.npy', label)

a = np.load('测试数据/G10data.npy')
b = np.load('测试数据/G10label.npy')
print(a.shape)
print(b[:10])
