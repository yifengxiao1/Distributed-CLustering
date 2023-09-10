import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from HashcodeCluster import spectral_clustering
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as MI
from sklearn.cluster import KMeans
from scipy.io import loadmat
import spectral as spy

class Linear(nn.Module):
    def __init__(self, code_len):
        super(Linear, self).__init__()
        self.code_len = code_len
        # self.conv1 = nn.Sequential(  # input_size=(1*32*16)
        #     nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 保证输入输出尺寸相同
        #     nn.ReLU(),  # input_size=(6*32*16)
        #     nn.MaxPool2d(kernel_size=2, stride=2), )  # output_size=(6*16*8)
        # self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5),
        #                            nn.ReLU(),  # input_size=(16*12*4)
        #                            nn.MaxPool2d(2, 2))  # output_size=(16*6*2) )
        self.fc1 = nn.Sequential(nn.Linear(36, 36), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(36, self.code_len), nn.Tanh())

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.size()[0], -1)
        x = self.fc2(self.fc1(x))
        return x

    def clistering_forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.size()[0], -1)
        x = torch.sign(self.fc2(self.fc1(x)))
        return x

def clust(epoch):
    # test_data = images[epoch * batch_size * sites: (epoch + 1) * batch_size * sites, :]
    # True_labels = labels[epoch * batch_size * sites: (epoch + 1) * batch_size * sites]
    test_data = images
    True_labels = labels
    with torch.no_grad():  # 聚类中不需要反向传播
        hash_codes = model.clistering_forward(
            torch.tensor(test_data).to(torch.float))  # test_data.reshape(len(True_labels), -1)
    # if torch.unique(hash_codes, dim=0).shape[0] < 9:
    #     print('第', epoch + 1, '轮: 码不够')
    #     return
    # km = KMeans(n_clusters=6).fit(np.array(hash_codes))
    # clust_labels = km.labels_
    clust_labels = SpectralClustering(n_clusters=6, gamma=0.1).fit_predict(np.array(hash_codes))
    # clust_codes = np.array(torch.unique(hash_codes, dim=0))
    # km = KMeans(n_clusters=10).fit(clust_codes)
    # nums = code2num(np.array(hash_codes))
    # clust_nums = code2num(clust_codes)
    # clust_labels = []
    # for i in range(len(nums)):
    #     clust_labels.append(km.labels_[np.where(clust_nums == nums[i])])
    # clust_labels = np.array(clust_labels).flatten()
    print('第', epoch + 1, '轮: MI=', MI(True_labels, clust_labels))
    Mi.append(MI(True_labels, clust_labels))
    # Purity
    purity = purity_score(clust_labels, True_labels)
    Purity.append(purity)
    print('第', epoch + 1, '轮: Purity=', purity)


from sklearn.metrics import accuracy_score
def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


dataset = loadmat('测试数据/satellite/归一化数据/minmax_satelite.mat')['minmax_scaling']
images = dataset[:, 1:].reshape(-1, 36).astype(np.float16)
labels = dataset[:, 0].reshape(-1).astype(np.float16)
model = torch.load('satellite/model/4.pth')
# sites = 100  # 实际上增加一次聚类的样本，想看看效果
# train_size = 10  # 从batchsize个数据中选trainsize个数据点参与训练
# model = torch.load('G9LeNet.pth')
Mi = list()
Purity = list()
clust(19)


