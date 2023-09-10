import numpy as np
from sklearn.cluster import SpectralClustering


def spectral_clustering(n_clusters, hashcode):
    # hashcode去重和统计权重
    temp = [tuple(t) for t in hashcode]
    code_dist = dict()
    for a in temp:
        code_dist[a] = temp.count(a)
        while a in temp:
            temp.remove(a)
        # print(len(temp))
    # np.save('code_dist.npy', code_dist)
    codes = list(code_dist.keys())
    adjacency_matrix = np.zeros([len(codes), len(codes)])
    # print(len(codes))
    for i in range(len(codes)):
        for j in range(i+1, len(codes)):
            a = np.array(codes[i])
            b = np.array(codes[j])
            adjacency_matrix[i][j] = adjacency_matrix[j][i] = (1 + a.dot(b) / len(codes[i])) * code_dist[codes[i]] * code_dist[codes[j]]  # 加不加权重好像对结果影响不大
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=100)
    sc.fit_predict(adjacency_matrix)
    label = np.zeros(len(hashcode))
    for k, code in enumerate(hashcode):
        label[k] = sc.labels_[codes.index(tuple(code))]
    return label
