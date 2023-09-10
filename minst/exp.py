import torch
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as MI

# data0, target = make_blobs(n_samples=1000000, n_features=50, centers=8)
# train_data = data0[:900000, :]
# test_data = data0[100000:101000, :]
# np.save('测试数据/train_data.npy', train_data)
# np.save('测试数据/test_data.npy', test_data)
# np.save('测试数据/target.npy', target)
images = np.load('sorted_images.npy')
labels = np.load('sort_labels.npy')
for i in range(4):
    images = np.append(images, images, axis=0)
    labels = np.append(labels, labels, axis=0)
# test_data = test_data[:500, :]
# target = np.load('测试数据/target.npy')
print(images.shape)

# tsne = TSNE(n_components=3, learning_rate=100).fit_transform(data)
# fig = plt.figure()
# ax = Axes3D(fig)
# colors = ['black', 'tomato', 'yellow', 'cyan', 'blue', 'lime', 'm', 'peru']
# for i in range(len(colors)):
#     for j in range(len(target)):
#         if target[j] == i:
#             ax.scatter(tsne[j, 0], tsne[j, 1], tsne[j, 2], c=colors[i])
#
# plt.savefig('123.png')
# fig.show()


def BN(raw_data):
    a, b = raw_data.size()
    feature_max = torch.max(raw_data, dim=0)
    feature_min = torch.min(raw_data, dim=0)
    for i in range(a):
        for j in range(b):
            raw_data[i][j] = (data[i][j] - feature_min[0][j]) / (feature_max[0][j] - feature_min[0][j])
    return raw_data

def getdist(data, batch_size):
    global data_dist
    length = batch_size
    for i in range(length):
        for j in range(i+1, length):
            if i == 0 and j == 1:
                data_dist = (data[i] - data[j]).unsqueeze(0)
            else:
                data_dist = torch.cat([data_dist, (data[i] - data[j]).unsqueeze(0)], 0)
    return data_dist


UE = 100
batch_size = 3
epoch = 1000
# bn = torch.nn.BatchNorm1d(50)
x = Variable(torch.rand((784, 32))-0.5, requires_grad=True)
# print(x)
# bias = Variable(torch.rand(128), requires_grad=True)
gradients = torch.zeros(UE, 784, 32)
all_loss = torch.zeros(UE)
lr = 1
Accuracy = []
bs_Accuracy = []
loss1 = []
for j in range(epoch):
    hash_codes = list()
    hash_codes_string = list()
    original_codes = list()
    for i in range(UE):
        #  训练
        data = torch.tensor(images[300 * j + 3 * i:300 * j + 3 * i + 3, :].astype(np.float32))
        data = data/255
        data_dist0 = torch.stack((data[0] - data[1], data[0] - data[2], data[1] - data[2]))
        data_dist = getdist(data, 3)
        # print(data_dist.equal(data_dist0))  # 明明二者相等，为什么0有效，非0无效？
        y = torch.tanh(torch.matmul(data, x)) / 2
        hash_dist0 = torch.stack((y[0] - y[1], y[0] - y[2], y[1] - y[2]))
        hash_dist = getdist(y, 3)
        # print(hash_dist.equal(hash_dist0))
        # print('原空间距离：', torch.norm(data_dist, dim=1), 'hash空间距离：', torch.norm(hash_dist, p=1, dim=1))
        # loss = torch.norm(torch.norm(data_dist0, dim=1) - torch.norm(hash_dist, p=1, dim=1), p=1) / 3 # + torch.norm(x)  # 用原空间与hash空间距离差为loss
        loss = torch.norm(torch.mul(torch.exp(-torch.norm(data_dist0, dim=1)), torch.pow(torch.norm(hash_dist0, p=1, dim=1), 2)))
        all_loss[i] = loss
        x.retain_grad()
        # bias.retain_grad()
        loss.backward()
        gradients[i] = x.grad
        #  哈希码聚类
        hash_code = torch.sgn(torch.matmul(data[2], x))
        # print(hash_code)
        original_code = data[2]
        hash_codes.append(hash_code.tolist())
        hash_code_string = list((map(str, hash_code.tolist())))
        hash_codes_string.append(''.join(hash_code_string))
        original_codes.append(original_code.tolist())
    mean_loss = torch.mean(all_loss)
    # mean_loss.backward()
    a = mean_loss.float()
    loss1.append(a.detach().cpu().numpy())
    grad = torch.mean(gradients, dim=0)
    # print(grad)
    if j < 500:
        x = x - lr * grad / (j + 1)
    else:
        x = x - lr * grad / (250 * (1.01 ** (j - 250)))
    print('第', j, '轮：', mean_loss)
    print(x)
    true_labels = labels[j * 300 + 2:(j + 1) * 300 + 2:3]
    length1 = len(set(true_labels))
    length2 = len(set(hash_codes_string))
    print(length1, length2)
    if length1 == 10 and length2 > 10:
        accuracy = 0
        km = KMeans(n_clusters=10).fit(np.array(hash_codes))
        clust_labels = km.labels_
        # print(true_labels)
        # print(clust_labels)
        for k in range(10):
            k_indexes = np.argwhere(clust_labels == k)
            k_labels = true_labels[k_indexes]
            k_labels = np.array(k_labels).flatten()
            d = np.argmax(np.bincount(k_labels))  # 出现次数最多的元素
            e = np.bincount(k_labels)[d]  # 出现次数
            accuracy += e / 100
        print(accuracy)
        Accuracy.append(accuracy)
        # 基线
        bs_accuracy = 0
        bs_km = KMeans(n_clusters=10).fit(np.array(original_codes))
        bs_clust_labels = bs_km.labels_
        # print(true_labels)
        # print(clust_labels)
        for k in range(10):
            k_indexes = np.argwhere(bs_clust_labels == k)
            k_labels = true_labels[k_indexes]
            k_labels = np.array(k_labels).flatten()
            d = np.argmax(np.bincount(k_labels))  # 出现次数最多的元素
            e = np.bincount(k_labels)[d]  # 出现次数
            bs_accuracy += e / 100
        print(bs_accuracy)
        bs_Accuracy.append(bs_accuracy)
    else:
        print('此轮聚类无效')
np.save('实验结果/32Accuracy.npy', Accuracy)
np.save('实验结果/32bs_Accuracy.npy', bs_Accuracy)
print(loss1)
np.save('实验结果/32loss.npy', loss1)
# from matplotlib.ticker import FuncFormatter
# for i in range(100):
#     mimax = np.max(mi[i*10:i*10+9])
# plt.plot(np.array(mi[800:1000]))
# plt.xlabel('训练轮数')
# plt.ylabel('聚类准确率')
# def to_percent(temp, position):
#   return '%1.0f'%(10*temp) + '%'
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.show()

# torch.save(x, '测试数据/x')
#
#
# # test
# x = torch.load('测试数据/x').cuda()
# data = bn(torch.tensor(test_data).to(torch.float32).cuda())
# hashed_data = torch.sgn(torch.matmul(data, x))
# dist_matrix = torch.zeros([500, 500])
# for i in range(len(test_data)):
#     for j in range(i+1, len(test_data)):
#         dist_matrix[i][j] = dist_matrix[j][i] = torch.norm(hashed_data[i]-hashed_data[j], p=1)/2
#     print(i)
# torch.save(dist_matrix, '测试数据/dist_matrix')
#
