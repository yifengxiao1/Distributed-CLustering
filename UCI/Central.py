import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as MI
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

a = loadmat('测试数据/satellite/归一化数据/minmax_satelite.mat')['minmax_scaling']
label = a[:, 0]
data = a[:, 1:]


ypred = SpectralClustering(n_clusters=6, gamma=0.1).fit_predict(data)
# ypred = KMeans(n_clusters=6).fit_predict(data)
# np.save('实验结果/musk2/centralkmeans.npy',ypred)


print(MI(label,ypred), purity_score(label,ypred))


