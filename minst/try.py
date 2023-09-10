import numpy as np
from mnist import load_mnist
from PIL import Image

a = 10
x_train = np.load('测试数据\'
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
im = Image.new('L', (28 * a, 28 * a))  # 生成一个大的纯黑灰度图作为容器 长*宽

labels = []
images = []

for i in range(100):  # 从0开始 取100个整数
    img = x_train[i]
    label = t_train[i]
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))  # 数据格式转换为8位
    labels.append(label)
    images.append(pil_img)

# 在大的纯黑图灰度图上放单张图 10*10 共100张 得到100个数字的单张图
for k in range(10, 110, 10):
    for j in range(10000):
        if j < 10:
            print(k / 10)
            im.paste(images[j], (j * 28, 28 * (int(k / 10) - 1)))
        elif j < k:
            im.paste(images[j], ((j - k + 10) * 28, 28 * (int(k / 10) - 1)))

im.show()  # 显示图像
print(labels)es[0]==wholecodes[1])