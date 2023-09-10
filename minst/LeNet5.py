import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as MI


class LeNet(nn.Module):
    def __init__(self, code_len):
        super(LeNet, self).__init__()
        self.code_len = code_len
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2), )  # output_size=(6*14*14)
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),  # input_size=(16*10*10)
                                   nn.MaxPool2d(2, 2))  # output_size=(16*5*5) )
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, self.code_len), nn.Tanh())

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
                   (epoch * sites * batch_size + site * batch_size) % 60000:(
                                                                                    epoch * sites * batch_size + site * batch_size + batch_size) % 60000 \
                       if (epoch * sites * batch_size + site * batch_size + batch_size) % 60000 != 0 else 60000,
                   :] / 255 - 0.5  # 手动读数据
            # train_data = torch.tensor(data_selection(data, train_size))
            train_data = torch.tensor(data)
            data_dist = getdist(train_data, train_size).to(torch.float)
            # data_dist = getcosine(train_data, train_size).to(torch.float)
            # data_dist = torch.stack((train_data[0] - train_data[1], train_data[0] - train_data[2], train_data[1] - train_data[2]))
            train_data = train_data.unsqueeze(1).reshape(train_size, 1, 28, 28).to(torch.float)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model.forward(train_data)  # 得到预测值
            output_dist = getdist(output, train_size).to(torch.float)
            # output_dist = getcosinehash(output, train_size).to(torch.float)
            # output_dist = torch.stack((output[0] - output[1], output[0] - output[2], output[1] - output[2]))
            loss = torch.norm(a * torch.norm(data_dist, dim=1) - torch.norm(output_dist, p=1, dim=1), p=1) / (
                    train_size * (train_size - 1) / 2)  # 用原空间与hash空间距离差为loss
            # loss = torch.norm(data_dist-output_dist)
            # loss = torch.norm(
            #     torch.mul(torch.exp(-0.001*torch.pow(torch.norm(data_dist, dim=1), 2)), torch.pow(torch.norm(output_dist, p=1, dim=1), 2)))
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss
        train_loss /= train_size
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        torch.save(model, '实验结果/model/'+'{a}'.format(a=epoch + 1)+'.pth')
        clust(epoch)


def clust(epoch):
    # test_data = torch.tensor(images[(epoch * sites * batch_size) % 70000: ((epoch + 1) * sites * batch_size) % 70000 \
    #     if ((epoch + 1) * sites * batch_size) % 70000 != 0 else 70000, :])
    # True_labels = labels[(epoch * sites * batch_size) % 70000: ((epoch + 1) * sites * batch_size) % 70000 \
    #     if ((epoch + 1) * sites * batch_size) % 70000 != 0 else 70000]
    test_data = torch.tensor(images)
    True_labels = torch.tensor(labels)
    # with torch.no_grad():  # 训练集中不需要反向传播
    #     hash_codes = model.clistering_forward(
    #         test_data.reshape(-1, 28, 28).unsqueeze(1).to(torch.float))
    # km = KMeans(n_clusters=10).fit(np.array(hash_codes))
    # clust_labels = km.labels_
    # print('第', epoch + 1, '轮: MI=', MI(True_labels, clust_labels))
    # Mi.append(MI(True_labels, clust_labels))
    # # Purity
    # purity = purity_score(True_labels, clust_labels)
    # Purity.append(purity)
    # print('第', epoch + 1, '轮: Purity=', purity)
    codes = []
    for i in range(test_data.shape[0]):
        input = test_data[i]
        with torch.no_grad():
            output = np.array(model.clistering_forward(
                input.reshape(-1, 28, 28).unsqueeze(1).to(torch.float)))
            output.resize(16)
            output[output == -1] = 0
            codes.append(''.join(str(i)[0] for i in output))
    num_codes = len(list(set(codes)))
    print('第', epoch + 1, '轮: Cost=', num_params * 32 * 2*(epoch+1) * sites + (32+code_len) * num_codes)


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
    images = np.load('测试数据/images.npy')[:60000,]
    labels = np.load('测试数据/labels.npy')[:60000]
    a = 1
    code_len = 16
    n_epochs = 500
    sites = 10
    # sites = 100  # 实际上增加一次聚类的样本，想看看效果
    batch_size = 10
    train_size = 10  # 从batchsize个数据中选trainsize个数据点参与训练
    model = torch.load('实验结果/16bit_model.pth')
    # model = LeNet(code_len)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Mi = list()
    Purity = list()
    clust(0)
    # train()
    # np.save('实验结果/10bit_MI.npy', Mi)
    # np.save('实验结果/10bit_Purity.npy', Purity)
