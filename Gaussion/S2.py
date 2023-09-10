import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from HashcodeCluster import spectral_clustering
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as MI
from sklearn.cluster import KMeans



class LeNet(nn.Module):
    def __init__(self, code_len):
        super(LeNet, self).__init__()
        self.code_len = code_len
        self.conv1 = nn.Sequential(  # input_size=(1*16*8)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*16*8)
            nn.MaxPool2d(kernel_size=2, stride=2), )  # output_size=(6*8*4)
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 3),
                                   nn.ReLU(),  # input_size=(16*6*2)
                                   nn.MaxPool2d(2, 2))  # output_size=(16*3*1) )
        self.fc1 = nn.Sequential(nn.Linear(16 * 3 * 1, self.code_len), nn.Tanh())

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

def getcosine(data, train_size):
    global data_dist
    length = train_size
    for i in range(length):
        for j in range(i + 1, length):
            if i == 0 and j == 1:
                data_dist = (torch.dot(data[i], data[j])/(torch.norm(data[i]) * torch.norm(data[j]))).unsqueeze(0)
            else:
                data_dist = torch.cat([data_dist, (torch.dot(data[i], data[j])/(torch.norm(data[i]) * torch.norm(data[j]))).unsqueeze(0)], 0)
    return data_dist

def getcosinehash(data, train_size):
    global data_dist
    length = train_size
    for i in range(length):
        for j in range(i + 1, length):
            if i == 0 and j == 1:
                data_dist = (torch.dot(data[i], data[j])/code_len).unsqueeze(0)
            else:
                data_dist = torch.cat([data_dist, (torch.dot(data[i], data[j])/code_len).unsqueeze(0)], 0)
    return data_dist


def train():
    # 定义损失函数和优化器
    # lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for site in range(sites):
            data = images[
                   (epoch * sites * batch_size + site * batch_size) % 350000:(
                                                                                    epoch * sites * batch_size + site * batch_size + batch_size) % 350000 \
                       if (epoch * sites * batch_size + site * batch_size + batch_size) % 350000 != 0 else 350000,
                   :] / 255  # 手动读数据
            # train_data = torch.tensor(data_selection(data, train_size))
            train_data = torch.tensor(data)
            data_dist = getdist(train_data, train_size).to(torch.float)
            # data_dist = getcosine(train_data, train_size).to(torch.float)
            # data_dist = torch.stack((train_data[0] - train_data[1], train_data[0] - train_data[2], train_data[1] - train_data[2]))
            train_data = train_data.unsqueeze(1).reshape(train_size, 1, 16, 8).to(torch.float)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model.forward(train_data)  # 得到预测值
            output_dist = getdist(output, train_size).to(torch.float)
            # output_dist = getcosinehash(output, train_size).to(torch.float)
            # output_dist = torch.stack((output[0] - output[1], output[0] - output[2], output[1] - output[2]))
            loss = torch.norm(torch.norm(data_dist, dim=1) - torch.norm(output_dist, p=1, dim=1), p=1) / (
                    train_size * (train_size - 1) / 2)  # 用原空间与hash空间距离差为loss
            # loss = torch.norm(data_dist-output_dist)
            # loss = torch.norm(
            #     torch.mul(torch.exp(-0.001*torch.pow(torch.norm(data_dist, dim=1), 2)), torch.pow(torch.norm(output_dist, p=1, dim=1), 2)))
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss
        train_loss /= train_size
        Loss.append(float(train_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        clust(epoch)


def clust(epoch=0):
    # test_data = torch.tensor(images[(epoch * sites * batch_size) % 350000: ((epoch + 1) * sites * batch_size) % 350000 \
    #     if ((epoch + 1) * sites * batch_size) % 350000 != 0 else 350000, :])
    # True_labels = labels[(epoch * sites * batch_size) % 350000: ((epoch + 1) * sites * batch_size) % 350000 \
    #     if ((epoch + 1) * sites * batch_size) % 350000 != 0 else 350000]
    # with torch.no_grad():  # 训练集中不需要反向传播
    #     hash_codes = model.clistering_forward(
    #         test_data.reshape(sites * batch_size, 16, 8).unsqueeze(1).to(torch.float))
    # km = KMeans(n_clusters=140).fit(np.array(hash_codes))
    # clust_labels = km.labels_
    # print('第', epoch + 1, '轮: MI=', MI(True_labels, clust_labels))
    # Mi.append(MI(True_labels, clust_labels))
    # # Purity
    # purity = pur(clust_labels, True_labels)
    # Purity.append(purity)
    # print('第', epoch + 1, '轮: Purity=', purity)
    codes = np.load('实验结果/S2/64bitcodes.npy')
    codes = list(set(codes))
    num_codes = len(codes)
    print('第', epoch + 1, '轮: Cost=', num_params * 32 * 41 * sites + (32 + code_len) * num_codes)



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



def data_selection(data, size):
    """
从数据中选取间距较大的size个点，作为训练数据
    :param data: 待搜索数据
    :param size: 目标batch_size
    """
    data_size = len(data)
    indexes = []
    indexes.append(int(np.random.uniform(0, data_size)))
    for i in range(1, size):
        dist = 0
        for index in indexes:
            dist += np.linalg.norm(data - data[index], axis=1)
        new_index = np.argmax(dist)
        indexes.append(new_index)
    return data[indexes]


if __name__ == '__main__':
    # images = np.load('sorted_images.npy')
    # labels = np.load('sort_labels.npy')
    images = np.load('测试数据/S2data.npy')
    labels = np.load('测试数据/S2label.npy')
    code_len = 64
    n_epochs = 5
    sites = 20
    # sites = 100  # 实际上增加一次聚类的样本，想看看效果
    batch_size = 10
    train_size = 10
    # model = LeNet(code_len)
    model = torch.load('S2LeNet.pth')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Mi = list()
    Purity = list()
    Loss = list()
    # train()
    clust()
    # np.save('实验结果/S2/64bit_Loss_60.npy', Loss)
    # torch.save(model, 'S2LeNet.pth')
    # np.save('实验结果/S2/64bit_MI.npy', Mi)
    # np.save('实验结果/S2/64bit_Purity.npy', Purity)








