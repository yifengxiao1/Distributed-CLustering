import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral as spy


# 获取mat格式的数据，loadmat输出的是dict，所以需要进行定位
input_image = loadmat('data set/Pavia.mat')['pavia']
gt = loadmat('data set/Pavia_gt.mat')['pavia_gt']


view1 = spy.imshow(data=input_image, bands=[69, 27, 11], title="img")  # 图像显示

view2 = spy.imshow(classes=gt, title="gt")  # 地物类别显示

view3 = spy.imshow(data=input_image, bands=[69, 27, 11], classes=gt)
view3.set_display_mode("overlay")
view3.class_alpha = 0.3  # 设置类别透明度为0.3



# pc = spy.principal_components(input_image)  # N维特征显示 view_nd与view_cube需要ipython 命令行输入：ipython --pylab
# xdata = pc.transform(input_image)  # 把数据转换到主成分空间
# spy.view_nd(xdata[:, :, :15], classes=gt)

plt.pause(60)

