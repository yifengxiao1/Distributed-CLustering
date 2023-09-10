import numpy as np
import random
import sys

images = np.load('测试数据/images.npy')
labels = np.load('测试数据/labels.npy')
new_images = []
new_labels = []
for i in range(6000):
    print(i)
    for j in range(10):
        indexes = np.where(labels == j)
        index = indexes[0][i]
        new_images.append(images[index])
        new_labels.append(labels[index])
    # print('sorted: ', sys.getsizeof(new_images)/1024/1024, 'MB')
    # print('original: ', sys.getsizeof(images)/1024/1024, 'MB')

np.save('sorted_images.npy', new_images)
np.save('sort_labels.npy', new_labels)
print(new_images.shape)
