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
        self.fc1 = nn.Sequential(nn.Linear(204, 204), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(204, self.code_len), nn.Tanh())

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


def getdist(data, train_size):
    global data_dist
    length = train_size
    for i in range(length):
        for j in range(i + 1, length):
            if i == 0 and j == 1:
                data_dist = (data[i] - data[j]).unsqueeze(0)
            else:
                data_dist = torch.cat([data_dist, (data[i] - data[j]).unsqueeze(0)], 0)
    return data_dist

# def getcosine(data, train_size):
#     global data_dist
#     length = train_size
#     for i in range(length):
#         for j in range(i + 1, length):
#             if i == 0 and j == 1:
#                 data_dist = (torch.dot(data[i], data[j])/(torch.norm(data[i]) * torch.norm(data[j]))).unsqueeze(0)
#             else:
#                 data_dist = torch.cat([data_dist, (torch.dot(data[i], data[j])/(torch.norm(data[i]) * torch.norm(data[j]))).unsqueeze(0)], 0)
#     return data_dist
#
# def getcosinehash(data, train_size):
#     global data_dist
#     length = train_size
#     for i in range(length):
#         for j in range(i + 1, length):
#             if i == 0 and j == 1:
#                 data_dist = (torch.dot(data[i], data[j])/code_len).unsqueeze(0)
#             else:
#                 data_dist = torch.cat([data_dist, (torch.dot(data[i], data[j])/code_len).unsqueeze(0)], 0)
#     return data_dist


def train():
    # 定义损失函数和优化器
    # lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for site in range(sites):
            data = images[
                   (epoch * sites * batch_size + site * batch_size):(epoch * sites * batch_size + site * batch_size + batch_size),:]
            # train_data = torch.tensor(data_selection(data, train_size))
            train_data = torch.tensor(data).to(torch.float)
            data_dist = getdist(train_data, batch_size).to(torch.float)
            # data_dist = getcosine(train_data, train_size).to(torch.float)
            # data_dist = torch.stack((train_data[0] - train_data[1], train_data[0] - train_data[2], train_data[1] - train_data[2]))
            # train_data = train_data.unsqueeze(1).reshape(batch_size, -1).to(torch.float)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model.forward(train_data)  # 得到预测值
            output_dist = getdist(output, batch_size).to(torch.float)
            # output_dist = getcosinehash(output, train_size).to(torch.float)
            # output_dist = torch.stack((output[0] - output[1], output[0] - output[2], output[1] - output[2]))
            loss = torch.norm(torch.norm(data_dist, dim=1) * a - torch.norm(output_dist, p=1, dim=1), p=1) / (
                    batch_size * (batch_size - 1) / 2) - torch.norm(torch.norm(output_dist, p=1, dim=1), p=1) / (
                    batch_size * (batch_size - 1) / 2)   # 用原空间与hash空间距离差为loss
            # print(torch.norm(torch.norm(output_dist, p=1, dim=1), p=1))
            # loss = torch.norm(data_dist-output_dist)
            # loss = torch.norm(
            #     torch.mul(torch.exp(-0.001*torch.pow(torch.norm(data_dist, dim=1), 2)), torch.pow(torch.norm(output_dist, p=1, dim=1), 2)))
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss
        train_loss /= batch_size
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        torch.save(model, 'Results/Salinas/model/'+'{a}'.format(a=epoch + 1)+'.pth')
        clust(epoch)



def clust(epoch=0):
    # test_data = images[epoch * batch_size * sites: (epoch + 1) * batch_size * sites, :]
    # True_labels = labels[epoch * batch_size * sites: (epoch + 1) * batch_size * sites]
    test_data = images
    True_labels = labels
    codes = []
    for i in range(test_data.shape[0]):
        input = test_data[i]
        with torch.no_grad():
            output = np.array(model.clistering_forward(
                torch.tensor(input).to(torch.float)))
            output.resize(24)
            output[output == -1] = 0
            codes.append(''.join(str(i)[0] for i in output))
    num_codes = len(list(set(codes)))
    # Spe = SpectralClustering(n_clusters=9, gamma=0.1).fit(np.array(data))
    # clust_labels = Spe.labels_
    print('第', epoch + 1, '轮: Cost=', num_params * 32 * 2*(epoch+1) * sites + (32 + code_len) * num_codes)
    # with torch.no_grad():  # 聚类中不需要反向传播
    #     hash_codes = model.clistering_forward(
    #         torch.tensor(test_data).to(torch.float))  # test_data.reshape(len(True_labels), -1)
    # # if torch.unique(hash_codes, dim=0).shape[0] < 9:
    # #     print('第', epoch + 1, '轮: 码不够')
    # #     return
    # km = KMeans(n_clusters=16).fit(np.array(hash_codes))
    # clust_labels = km.labels_
    # clust_labels= spectral_clustering(n_clusters=9, hashcode=hash_codes)
    # clust_codes = np.array(torch.unique(hash_codes, dim=0))
    # km = KMeans(n_clusters=10).fit(clust_codes)
    # nums = code2num(np.array(hash_codes))
    # clust_nums = code2num(clust_codes)
    # clust_labels = []
    # for i in range(len(nums)):
    #     clust_labels.append(km.labels_[np.where(clust_nums == nums[i])])
    # clust_labels = np.array(clust_labels).flatten()
    # print('第', epoch + 1, '轮: MI=', MI(True_labels, clust_labels))
    # Mi.append(MI(True_labels, clust_labels))
    # # Purity
    # purity = purity_score(clust_labels, True_labels)
    # Purity.append(purity)
    # print('第', epoch + 1, '轮: Purity=', purity)

def code2num(codes):
    """conde2num
            Args:
                codes(np.array): code_number * code_len

            Returns:
                nums(np.array): binary number to float
        """
    nums = []
    for i in range(codes.shape[0]):
        num = 0
        for j in range(len(codes[i])):
            if codes[i, j] == 1:
                num += 2 ** j
        nums.append(num)
    return np.array(nums)



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

from sklearn.metrics import accuracy_score
import numpy as np

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




# def data_selection(data, size):
#     """
# 从数据中选取间距较大的size个点，作为训练数据
#     :param data: 待搜索数据
#     :param size: 目标batch_size
#     """
#     data_size = len(data)
#     indexes = []
#     indexes.append(int(np.random.uniform(0, data_size)))
#     for i in range(1, size):
#         dist = 0
#         for index in indexes:
#             dist += np.linalg.norm(data - data[index], axis=1)
#         new_index = np.argmax(dist)
#         indexes.append(new_index)
#     return data[indexes]


if __name__ == '__main__':
    # images = np.load('sorted_images.npy')
    # labels = np.load('sort_labels.npy')
    images = loadmat('data set/Salinas_corrected.mat')['salinas_corrected'].reshape(-1, 204).astype(np.float16)
    images = images / np.max(images) -0.5
    labels = loadmat('data set/Salinas_gt.mat')['salinas_gt'].reshape(-1).astype(np.float16)

    zero_indexes = np.where(labels == 0)
    labels = np.delete(labels, zero_indexes)
    images = np.delete(images, zero_indexes, axis=0)
    a = 1
    code_len = 24
    n_epochs = 100
    sites = 10
    batch_size = 5
    # train_size = 10  # 从batchsize个数据中选trainsize个数据点参与训练
    # model = Linear(code_len)
    model = torch.load('Results/Salinas/model/204_204_24.pth')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Mi = list()
    Purity = list()
    # train()
    clust()
    # torch.save(model, 'Results/Salinas/model/8bit.pth')
    # np.save('Results/Salinas/13bit_MI.npy', Mi)
    # np.save('Results/Salinas/13bit_Purity.npy', Purity)








