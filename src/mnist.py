import numpy as np  # 导入NumPy库，用于科学计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和分析
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于数据可视化

# import matplotlib.image as mping  # 导入Matplotlib的image模块，用于图像处理
import math  # 导入数学库，提供数学函数

# 从multilayer_perceptron.py文件导入MultilayerPerceptron类
from multilayer_perceptron import MultilayerPerceptron

# 使用pandas读取CSV文件，CSV包含手写数字图像数据
data = pd.read_csv("src/data/mnist-demo.csv")  # 读取CSV文件，CSV包含手写数字图像数据

# 以下代码用于显示MNIST数据集中的一些图像样本
numbers_to_display = 25  # 要显示的图像数量
num_cells = math.ceil(math.sqrt(numbers_to_display))  # 计算网格大小，向上取整
plt.figure(figsize=(10, 10))  # 创建一个10x10大小的图形

# 循环处理要显示的每个数字图像
for plot_index in range(numbers_to_display):
    # 从数据集中获取一个样本
    digit = data[plot_index : plot_index + 1].values
    # 提取标签（这个数字是几，例如0,1,2...9）
    digit_label = digit[0][0]
    # 提取像素值（图像数据，除了第一列标签外的所有数据）
    digit_pixels = digit[0][1:]
    # 计算图像的边长（MNIST图像是正方形的）
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    # 将一维像素数组重塑为二维图像矩阵
    frame = digit_pixels.reshape((image_size, image_size))
    # 在图形中创建子图
    plt.subplot(num_cells, num_cells, plot_index + 1)
    # 显示图像，使用灰度颜色映射
    plt.imshow(frame, cmap="Greys")
    # 设置图像标题为数字标签
    plt.title(digit_label)

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# 显示图形
plt.show()

# 数据集分割为训练集和测试集
# 随机抽取80%的数据作为训练集
train_data = data.sample(frac=0.8)
# 剩余20%的数据作为测试集（从原始数据中删除训练集的行）
test_data = data.drop(train_data.index)

# 将DataFrame转换为NumPy数组，便于后续处理
train_data = train_data.values
test_data = test_data.values

# 限制训练样本数量为5000，减少计算量
num_training_examples = 5000

# 准备训练数据
# x_train包含所有特征（像素值），不包括第一列的标签
x_train = train_data[:num_training_examples, 1:]
# y_train只包含标签（第一列）
y_train = train_data[:num_training_examples, [0]]

# 准备测试数据
# x_test包含所有特征（像素值），不包括第一列的标签
x_test = test_data[:, 1:]
# y_test只包含标签（第一列）
y_test = test_data[:, [0]]

# 神经网络层结构定义
# 784: 输入层神经元数量（28x28=784个像素）
# 25: 隐藏层神经元数量
# 10: 输出层神经元数量（对应0-9这10个数字）
layers = [784, 25, 10]

# 是否对数据进行归一化处理
normalize_data = True
# 最大迭代次数（训练过程中的梯度下降步数）
max_iterations = 500
# 学习率，控制每次梯度下降的步长
alpha = 0.1

# 创建多层感知机模型实例
# 参数包括：训练特征、训练标签、网络层结构、是否归一化数据
multilayer_perceptron = MultilayerPerceptron(x_train, y_train, layers, normalize_data)
# 训练模型，返回最终的参数（thetas）和每次迭代的代价（costs）
(thetas, costs) = multilayer_perceptron.train(max_iterations, alpha)

# 绘制代价函数随迭代次数的变化曲线
plt.plot(range(len(costs)), costs)
# 设置x轴标签
plt.xlabel("Gradient steps")
# 设置y轴标签（此处有笔误，应为ylabel）
plt.xlabel("costs")
# 显示图形
plt.show()

# 使用训练好的模型进行预测
# 预测训练集样本
y_train_predictions = multilayer_perceptron.predict(x_train)
# 预测测试集样本
y_test_predictions = multilayer_perceptron.predict(x_test)

# 计算训练集上的准确率
# np.sum(y_train_predictions == y_train)计算预测正确的样本数
# 除以总样本数再乘以100，得到百分比形式的准确率
train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
# 计算测试集上的准确率
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
# 打印准确率结果
print("训练集准确率：", train_p)
print("测试集准确率：", test_p)

# 可视化测试集上的预测结果
numbers_to_display = 64  # 要显示的图像数量

# 计算显示网格的尺寸
num_cells = math.ceil(math.sqrt(numbers_to_display))

# 创建一个大图形用于显示多个子图
plt.figure(figsize=(15, 15))

# 循环处理要显示的每个测试样本
for plot_index in range(numbers_to_display):
    # 获取真实标签
    digit_label = y_test[plot_index, 0]
    # 获取图像像素数据
    digit_pixels = x_test[plot_index, :]

    # 获取预测标签
    predicted_label = y_test_predictions[plot_index][0]

    # 计算图像尺寸（MNIST图像是28x28像素）
    image_size = int(math.sqrt(digit_pixels.shape[0]))

    # 将一维像素数组重塑为二维图像矩阵
    frame = digit_pixels.reshape((image_size, image_size))

    # 根据预测是否正确选择不同的颜色映射
    # 如果预测正确，使用绿色(Greens)；如果预测错误，使用红色(Reds)
    color_map = "Greens" if predicted_label == digit_label else "Reds"

    # 创建子图
    plt.subplot(num_cells, num_cells, plot_index + 1)
    # 显示图像，使用选定的颜色映射
    plt.imshow(frame, cmap=color_map)
    # 设置图像标题为预测的标签
    plt.title(predicted_label)
    # 关闭坐标轴刻度和标签
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.5, wspace=0.5)
# 显示图形
plt.show()
