import torch
import numpy as np
from G10 import  LeNet
from sklearn.metrics import adjusted_mutual_info_score as MI
from sklearn.cluster import KMeans

# images = torch.tensor(np.load('测试数据/G10data.npy'))
# labels = torch.tensor(np.load('测试数据/G10label.npy'))
# model = torch.load('G10LeNet.pth')
#
# codes = []
# for i in range(images.shape[0]):
#     input = images[i]
#     with torch.no_grad():
#         output = np.array(model.clistering_forward(
#                 input.reshape(-1, 32, 32).unsqueeze(1).to(torch.float)))
#         output.resize(64)
#         output[output == -1] = 0
#         codes.append(''.join(str(i)[0] for i in output))
#         print(i)
#
# np.save('实验结果/G10/64bitcodes.npy', np.array(codes))

wholecodes = np.load('实验结果/G10/64bitcodes.npy')
codes = list(set(wholecodes))
indexes = dict()
for i in range(len(codes)):
    index = np.argwhere(wholecodes==codes[i])
    indexes.update({i:index})


data = np.zeros([len(codes), len(codes[0])])
for i in range(len(codes)):
    for j in range(len(codes[i])):
        data[i,j] = int(codes[i][j])

km = KMeans(n_clusters=200).fit(np.array(data))
clust_labels = km.labels_
wholelabels= np.zeros(len(wholecodes))
for i in range(len(codes)):
    wholelabels[indexes[i]] = clust_labels[i]
labels = torch.tensor(np.load('测试数据/G10label.npy'))
print(MI(labels, wholelabels))

# np.save('实验结果/G10/64bitlabels.npy', wholelabels)




