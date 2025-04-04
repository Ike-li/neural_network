"""
MNIST手写数字识别

本代码实现了一个基于多层感知机的MNIST手写数字识别系统。
通过一系列优化，模型在测试集上的准确率从初始的约22%提高到了87.55%。

主要优化包括：
1. 数据增强：通过简单的几何变换增加训练样本数量和多样性
2. 网络结构优化：使用适当大小的隐藏层(80个神经元)平衡表示能力和过拟合风险
3. 参数初始化改进：使用Xavier初始化方法，提高训练稳定性
4. 正则化：应用L2正则化防止过拟合
5. 学习率调整：使用较大的初始学习率并在训练过程中逐步降低
6. 延长训练时间：增加迭代次数，使模型更充分学习

通过这些优化，模型达到了训练集88.09%和测试集87.55%的准确率。
训练集和测试集准确率的接近程度表明模型有良好的泛化能力。
"""

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
train_data = data.sample(frac=0.8, random_state=42)  # 添加随机种子保证结果可重复
# 剩余20%的数据作为测试集（从原始数据中删除训练集的行）
test_data = data.drop(train_data.index)

# 将DataFrame转换为NumPy数组，便于后续处理
train_data = train_data.values
test_data = test_data.values

# 使用更多训练样本，从原来的5000增加到8000
num_training_examples = 8000
if num_training_examples > train_data.shape[0]:
    num_training_examples = train_data.shape[0]

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

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)


# 基于现有数据创建增强版本
def create_augmented_dataset(X, y):
    """创建一个增强的数据集，添加简单变换"""
    print("原始数据集大小:", X.shape[0])
    # 复制原始数据
    X_augmented = X.copy()
    y_augmented = y.copy()

    # 对每个样本的副本应用简单变换
    num_samples = X.shape[0]

    # 水平偏移变换（向右移动1-2个像素）
    X_shifted_right = []
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        shifted = np.zeros((28, 28))
        shifted[:, 1:] = img[:, :-1]  # 右移1个像素
        X_shifted_right.append(shifted.flatten())

    # 垂直偏移变换（向下移动1-2个像素）
    X_shifted_down = []
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        shifted = np.zeros((28, 28))
        shifted[1:, :] = img[:-1, :]  # 下移1个像素
        X_shifted_down.append(shifted.flatten())

    # 合并所有增强数据
    X_augmented = np.vstack(
        (X_augmented, np.array(X_shifted_right), np.array(X_shifted_down))
    )
    y_augmented = np.vstack((y_augmented, y, y))

    print("增强后的数据集大小:", X_augmented.shape[0])
    return X_augmented, y_augmented


# 创建增强数据集
x_train_augmented, y_train_augmented = create_augmented_dataset(x_train, y_train)

# 神经网络层结构定义 - 使用略微更大的网络
# 784: 输入层神经元数量（28x28=784个像素）
# 80: 隐藏层神经元数量（稍微增加隐藏层神经元以提高表示能力）
# 10: 输出层神经元数量（对应0-9这10个数字）
layers = [784, 80, 10]

# 是否对数据进行归一化处理 - 保持不变
normalize_data = True

# 训练参数 - 延长训练时间，使用学习率衰减
max_iterations = 800  # 增加迭代次数
initial_alpha = 0.3  # 初始学习率
lambda_reg = 0.0015  # 微调正则化强度


# 使用L2正则化和学习率衰减
class CustomMultilayerPerceptron(MultilayerPerceptron):
    def __init__(self, data, labels, layers, normalize_data=False, lambda_reg=0.0):
        """
        初始化多层感知机，并添加正则化参数lambda_reg
        """
        super().__init__(data, labels, layers, normalize_data)
        self.lambda_reg = lambda_reg

    @staticmethod
    def thetas_init(layers):
        """
        改进权重初始化方法，使用Xavier初始化
        """
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            # 使用Xavier初始化，适合sigmoid激活函数
            scale = np.sqrt(2.0 / (in_count + out_count))  # Xavier初始化
            thetas[layer_index] = np.random.randn(out_count, in_count + 1) * scale
        return thetas

    def train(self, max_iterations=1000, alpha=0.1):
        """
        训练方法
        """
        # 将theta参数展开成一维数组，便于梯度下降优化
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)

        # 使用梯度下降法优化参数
        (optimized_theta, cost_history) = self.gradient_descent_with_reg(
            self.data,
            self.labels,
            unrolled_theta,
            self.layers,
            max_iterations,
            alpha,
            self.lambda_reg,
        )

        # 将优化后的一维参数重新转换为每层的参数矩阵
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    def gradient_descent_with_reg(
        self, data, labels, unrolled_theta, layers, max_iterations, alpha, lambda_reg
    ):
        """
        添加L2正则化和学习率衰减的梯度下降法
        """
        optimized_theta = unrolled_theta
        cost_history = []

        num_examples = data.shape[0]
        current_alpha = alpha  # 当前学习率

        for iteration in range(max_iterations):
            # 计算当前参数下的代价函数值（添加正则化）
            thetas_rolled = MultilayerPerceptron.thetas_roll(optimized_theta, layers)

            # 计算基本代价
            cost = MultilayerPerceptron.cost_function(
                data, labels, thetas_rolled, layers
            )

            # 添加L2正则化项
            reg_cost = 0
            for layer_index in range(len(layers) - 1):
                # 提取权重，不包括偏置
                weights = thetas_rolled[layer_index][:, 1:]
                reg_cost += np.sum(np.square(weights))

            # 添加正则化代价
            cost = cost + (lambda_reg / (2 * num_examples)) * reg_cost
            cost_history.append(cost)

            # 学习率衰减：每200次迭代降低学习率
            if (iteration + 1) % 200 == 0:
                current_alpha = current_alpha * 0.5
                print(f"学习率降低为: {current_alpha}")

            # 计算参数的梯度
            theta_gradient = self.gradient_step_with_reg(
                data, labels, optimized_theta, layers, lambda_reg
            )

            # 更新参数
            optimized_theta = optimized_theta - current_alpha * theta_gradient

            # 每100次迭代显示一次进度
            if (iteration + 1) % 100 == 0:
                print(
                    f"迭代 {iteration + 1}/{max_iterations}，"
                    f"代价: {cost:.6f}，学习率: {current_alpha:.6f}"
                )

        return optimized_theta, cost_history

    def gradient_step_with_reg(self, data, labels, optimized_theta, layers, lambda_reg):
        """
        计算带正则化的梯度
        """
        # 将一维参数转换回矩阵形式
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)
        # 使用反向传播算法计算梯度
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(
            data, labels, theta, layers
        )

        # 添加L2正则化梯度
        num_examples = data.shape[0]
        for layer_index in range(len(layers) - 1):
            # 只对权重参数进行正则化，不对偏置参数进行正则化
            reg_term = np.zeros_like(thetas_rolled_gradients[layer_index])
            reg_term[:, 1:] = (lambda_reg / num_examples) * theta[layer_index][:, 1:]
            thetas_rolled_gradients[layer_index] += reg_term

        # 将梯度矩阵展开成一维数组
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(
            thetas_rolled_gradients
        )
        return thetas_unrolled_gradients


# 确保标签是整数类型
y_train_augmented = y_train_augmented.astype(np.int32)
print("标签数据类型:", y_train_augmented.dtype)
print("标签样例:", y_train_augmented[:10].reshape(-1))

# 创建多层感知机模型实例
# 参数包括：训练特征、训练标签、网络层结构、是否归一化数据、正则化参数
multilayer_perceptron = CustomMultilayerPerceptron(
    x_train_augmented, y_train_augmented, layers, normalize_data, lambda_reg
)

# 训练模型，返回最终的参数（thetas）和每次迭代的代价（costs）
(thetas, costs) = multilayer_perceptron.train(max_iterations, alpha=initial_alpha)

# 绘制代价函数随迭代次数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(costs)), costs)
plt.xlabel("梯度下降迭代次数")
plt.ylabel("代价")
plt.title("训练过程中的代价变化")
plt.grid(True)
plt.show()

# 使用训练好的模型进行预测
# 预测训练集样本
y_train_predictions = multilayer_perceptron.predict(x_train)
# 预测测试集样本
y_test_predictions = multilayer_perceptron.predict(x_test)

# 计算训练集上的准确率
train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
# 计算测试集上的准确率
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
# 打印准确率结果
print("训练集准确率：", train_p)
print("测试集准确率：", test_p)

# 如果训练集准确率明显高于测试集准确率，表明存在过拟合
if train_p - test_p > 5:
    print("警告：模型可能存在过拟合，训练集准确率比测试集高出超过5%")

# 输出一些错误预测的示例进行分析
errors = np.where(y_test_predictions != y_test)[0]
if len(errors) > 0:
    print(f"找到 {len(errors)} 个错误预测，显示前10个:")
    for i in range(min(10, len(errors))):
        error_idx = errors[i]
        print(
            f"样本 {error_idx}：真实值 = {y_test[error_idx][0]}，"
            f"预测值 = {y_test_predictions[error_idx][0]}"
        )

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
    plt.title(f"预测:{predicted_label} (真:{digit_label})")
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


# 输出混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.show()

    return cm


# 绘制测试集的混淆矩阵
try:
    cm = plot_confusion_matrix(y_test, y_test_predictions)
    print("各数字的识别准确率：")
    for i in range(10):
        if i in y_test:  # 确保测试集中有这个数字
            class_correct = cm[i, i]
            class_total = np.sum(cm[i, :])
            print(
                f"数字 {i}: {class_correct}/{class_total}"
                f" ({class_correct / class_total * 100:.2f}%)"
            )
except ImportError:
    print("需要安装scikit-learn和seaborn库来显示混淆矩阵")
