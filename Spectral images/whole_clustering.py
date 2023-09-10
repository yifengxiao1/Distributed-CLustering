import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from HashcodeCluster import spectral_clustering
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as MI
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
def pur(cluster, label):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)

    return sum(np.max(count_all, axis=0))/len(cluster)


class LeNet(nn.Module):
    def __init__(self, code_len):
        super(LeNet, self).__init__()
        self.code_len = code_len
        self.conv1 = nn.Sequential(  # input_size=(1*32*16)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*32*16)
            nn.MaxPool2d(kernel_size=2, stride=2), )  # output_size=(6*16*8)
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),  # input_size=(16*12*4)
                                   nn.MaxPool2d(2, 2))  # output_size=(16*6*2) )
        self.fc1 = nn.Sequential(nn.Linear(16 * 6 * 2, self.code_len), nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)  # 试一试没有全联接，直接sgn？
        x = self.fc1(x)
        return x

    def clistering_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = torch.sign(self.fc1(x))
        return x


images = torch.tensor(np.load('测试数据/S1data.npy'))
labels = torch.tensor(np.load('测试数据/S1label.npy'))
model = torch.load('S1LeNet.pth')

codes = []
for i in range(images.shape[0]):
    input = images[i]
    with torch.no_grad():
        output = np.array(model.clistering_forward(
                input.reshape(-1, 8, 8).unsqueeze(1).to(torch.float)))
        output.resize(64)
        output[output == -1] = 0
        codes.append(''.join(str(i)[0] for i in output))
        print(i)
#
# np.save('实验结果/S1/64bitcodes.npy', np.array(codes))

wholecodes = np.load('实验结果/S1/64bitcodes.npy')
codes = list(set(wholecodes))
indexes = dict()
for i in range(len(codes)):
    index = np.argwhere(wholecodes==codes[i])
    indexes.update({i:index})


data = np.zeros([len(codes), len(codes[0])])
for i in range(len(codes)):
    for j in range(len(codes[i])):
        data[i,j] = int(codes[i][j])

# km = KMeans(n_clusters=120).fit(np.array(data))
# clust_labels = km.labels_
Spe = SpectralClustering(n_clusters=120, gamma=0.1).fit(np.array(data))
clust_labels = Spe.labels_
wholelabels= np.zeros(len(wholecodes))
for i in range(len(codes)):
    wholelabels[indexes[i]] = clust_labels[i]
labels = torch.tensor(np.load('测试数据/S1label.npy'))
print(MI(labels, wholelabels))
print(pur(labels, wholelabels))

# np.save('实验结果/G10/64bitlabels.npy', wholelabels)






