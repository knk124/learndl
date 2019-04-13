from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # 导入数据

# 选择网络的形状
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))  # 第一层，设置了神经元个数、激活函数和输入的大小
network.add(layers.Dense(10, activation='softmax'))  # 第二层，也是输出层
# 选择损失函数、优化器和度量结果好坏的指标
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255  # 将数据类型设置为float32，并使其值范围为0-1
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
# 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 拟合模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 在测试集上进行测试
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
# 显示数据
# digit = train_images[4]  # 显示第四个数字
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()