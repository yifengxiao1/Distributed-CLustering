import torch
import numpy as np
from HashcodeCluster import sectrual_clustering
from sklearn.metrics import adjusted_mutual_info_score as MI


class site(object):
    start = 0

    def __init__(self, model, data):
        self.local_model = model
        self.optimizer = torch.optim.SGD(params=self.local_model.parameters(), lr=0.01)
        self.local_data = data

    def train(self):
        # load training data
        batch_size = np.random.randint(10, 15)  # 有没有必要呢？没有
        train_data = self.local_data[self.start, self.start + batch_size]
        self.start += batch_size
        # train
        data_dist = getdist(train_data, batch_size)
        train_data = train_data.unsqueeze(1).reshape(batch_size, 1, 28, 28).to(torch.float)
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        output = self.local_model.forward(train_data)  # 得到预测值
        output_dist = getdist(output, batch_size).to(torch.float)
        loss = torch.norm(torch.norm(data_dist, dim=1) - torch.norm(output_dist, p=1, dim=1), p=1) / (
                batch_size * (batch_size - 1) / 2)
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()
        return self.local_model, batch_size

    def hashing(self):
        with torch.no_grad():  # 训练集中不需要反向传播, 所有本地数据参与聚类
            hash_codes = self.local_model.clistering_forward(
                self.local_data.reshape(-1, 28, 28).unsqueeze(1).to(torch.float))
        temp = [tuple(t) for t in hash_codes]
        code_dict = dict()
        for a in temp:
            code_dict[a] = temp.count(a)
            while a in temp:
                temp.remove(a)
        return code_dict


class cloud(object):
    def __init__(self, model):
        self.globel_model = model

    def clustering(self, hash_codes):
        clust_labels = sectrual_clustering(n_clusters=10, hashcode=hash_codes)
        # km = KMeans(n_clusters=10).fit(np.array(hash_codes))
        print('第', epoch + 1, '轮: MI=', MI(True_labels, clust_labels))
        Mi.append(MI(True_labels, clust_labels))
        # Purity
        purity = 0
        for k in range(10):
            k_indexes = np.argwhere(clust_labels == k)
            k_labels = True_labels[k_indexes.squeeze(1)]
            k_labels = np.array(k_labels).flatten()
            d = np.argmax(np.bincount(k_labels))  # 出现次数最多的元素
            e = np.bincount(k_labels)[d]  # 出现次数
            purity += e / len(True_labels)
        Purity.append(purity)
        print('第', epoch + 1, '轮: Purity=', purity)



def getdist(data, batch_size):
    global data_dist
    length = batch_size
    for i in range(length):
        for j in range(i + 1, length):
            if i == 0 and j == 1:
                data_dist = (data[i] - data[j]).unsqueeze(0)
            else:
                data_dist = torch.cat([data_dist, (data[i] - data[j]).unsqueeze(0)], 0)
    return data_dist