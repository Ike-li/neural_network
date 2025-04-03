import numpy as np  # 导入NumPy库，用于科学计算和数组操作

from src.sigmoid import Sigmoid
from src.training import Training


# 多层感知机类，实现了一个完整的神经网络
class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        """
        初始化多层感知机

        参数:
        data - 输入特征数据，形状为 (样本数, 特征数)
        labels - 标签数据，形状为 (样本数, 1)
        layers - 神经网络各层的神经元数量，例如 [784, 25, 10] 表示输入层784个神经元，隐藏层25个神经元，输出层10个神经元
        normalize_data - 是否对输入数据进行归一化处理
        """
        # 使用prepare_for_training函数预处理数据，返回处理后的数据和其他相关信息

        data_processed = Training.prepare(data, normalize_data=normalize_data)[0]
        self.data = data_processed  # 存储处理后的特征数据
        self.labels = labels  # 存储标签数据
        self.layers = layers  # 存储网络结构，例如 [784, 25, 10]
        self.normalize_data = normalize_data  # 存储是否归一化的标志
        # 初始化网络权重参数（theta）
        self.thetas = MultilayerPerceptron.thetas_init(layers)

    def predict(self, data):
        """
        使用训练好的模型进行预测

        参数:
        data - 要预测的特征数据，形状为 (样本数, 特征数)

        返回:
        predictions - 预测结果，形状为 (样本数, 1)，包含每个样本的预测类别（0-9）
        """
        # 对输入数据进行预处理，保持与训练数据相同的处理方式
        data_processed = Training.prepare(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]  # 获取样本数量

        # 使用前向传播计算预测结果
        predictions = MultilayerPerceptron.feedforward_propagation(
            data_processed, self.thetas, self.layers
        )

        # 返回预测类别，即输出层中概率最大的神经元对应的索引
        # np.argmax在axis=1上找最大值，相当于每个样本找出10个输出中最大的作为预测类别
        # reshape将结果整形为(样本数,1)的列向量
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):
        """
        训练神经网络模型

        参数:
        max_iterations - 最大迭代次数，即梯度下降的最大步数
        alpha - 学习率，控制每次参数更新的步长

        返回:
        thetas - 训练后的网络参数
        cost_history - 训练过程中的代价函数值历史记录
        """
        # 将theta参数展开成一维数组，便于梯度下降优化
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)

        # 使用梯度下降法优化参数
        (optimized_theta, cost_history) = MultilayerPerceptron.gradient_descent(
            self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha
        )

        # 将优化后的一维参数重新转换为每层的参数矩阵
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def thetas_init(layers):
        """
        初始化神经网络各层的参数矩阵

        参数:
        layers - 神经网络各层的神经元数量列表，例如 [784, 25, 10]

        返回:
        thetas - 字典，包含每层的参数矩阵
        """
        num_layers = len(layers)  # 获取网络层数
        thetas = {}  # 创建参数字典
        for layer_index in range(num_layers - 1):
            """
            会执行两次，得到两组参数矩阵：
            对于 [784, 25, 10] 的网络结构:
            第一层参数矩阵: 25×785 (25个隐藏层神经元，每个连接784个输入+1个偏置)
            第二层参数矩阵: 10×26 (10个输出层神经元，每个连接25个隐藏层输出+1个偏置)
            """
            in_count = layers[layer_index]  # 当前层神经元数量
            out_count = layers[layer_index + 1]  # 下一层神经元数量
            # 初始化参数矩阵，考虑偏置项，所以列数+1
            # 随机初始化参数，范围在0-0.05之间，很小的随机值有助于打破对称性
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """
        将每层的参数矩阵展开成一维数组，便于优化算法处理

        参数:
        thetas - 字典，包含每层的参数矩阵

        返回:
        unrolled_theta - 一维数组，包含所有参数
        """
        num_theta_layers = len(thetas)  # 获取网络层数-1（参数矩阵的数量）
        unrolled_theta = np.array([])  # 创建空数组用于存储展开后的参数
        for theta_layer_index in range(num_theta_layers):
            # 将每层的参数矩阵展平，并与之前的参数数组连接
            unrolled_theta = np.hstack(
                (unrolled_theta, thetas[theta_layer_index].flatten())
            )
        return unrolled_theta

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        """
        使用梯度下降法优化神经网络参数

        参数:
        data - 特征数据
        labels - 标签数据
        unrolled_theta - 展开的初始参数
        layers - 网络结构
        max_iterations - 最大迭代次数
        alpha - 学习率

        返回:
        optimized_theta - 优化后的参数（一维数组）
        cost_history - 训练过程中的代价函数值历史记录
        """
        optimized_theta = unrolled_theta  # 初始化优化参数为输入的初始参数
        cost_history = []  # 创建空列表用于记录代价函数值

        # 迭代max_iterations次，进行梯度下降优化
        for _ in range(max_iterations):
            # 计算当前参数下的代价函数值
            cost = MultilayerPerceptron.cost_function(
                data,
                labels,
                MultilayerPerceptron.thetas_roll(optimized_theta, layers),
                layers,
            )
            cost_history.append(cost)  # 记录代价函数值

            # 计算参数的梯度
            theta_gradient = MultilayerPerceptron.gradient_step(
                data, labels, optimized_theta, layers
            )
            # 更新参数：theta = theta - alpha * gradient
            optimized_theta = optimized_theta - alpha * theta_gradient

        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        """
        计算一步梯度下降的梯度

        参数:
        data - 特征数据
        labels - 标签数据
        optimized_theta - 当前的参数（一维数组）
        layers - 网络结构

        返回:
        thetas_unrolled_gradients - 展开的梯度（一维数组）
        """
        # 将一维参数转换回矩阵形式
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)
        # 使用反向传播算法计算梯度
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(
            data, labels, theta, layers
        )
        # 将梯度矩阵展开成一维数组
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(
            thetas_rolled_gradients
        )
        return thetas_unrolled_gradients

    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        """
        反向传播算法，计算神经网络参数的梯度

        参数:
        data - 特征数据，形状为 (样本数, 特征数)
        labels - 标签数据，形状为 (样本数, 1)
        thetas - 当前的参数字典
        layers - 网络结构

        返回:
        deltas - 各层参数的梯度字典
        """
        num_layers = len(layers)  # 网络层数
        (num_examples, num_features) = data.shape  # 样本数和特征数
        num_label_types = layers[-1]  # 输出类别数（10，对应数字0-9）

        deltas = {}  # 创建梯度字典
        # 初始化各层梯度为0矩阵
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]  # 当前层神经元数
            out_count = layers[layer_index + 1]  # 下一层神经元数
            deltas[layer_index] = np.zeros(
                (out_count, in_count + 1)
            )  # 形状与参数矩阵相同

        # 对每个样本执行一次前向传播和反向传播
        for example_index in range(num_examples):
            layers_inputs = {}  # 存储每层的净输入（加权和）
            layers_activations = {}  # 存储每层的激活值（经过激活函数后的输出）

            # 获取当前样本的特征，并重塑为列向量
            layers_activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = layers_activation  # 存储输入层的激活值（即特征值）

            # 前向传播，计算每层的净输入和激活值
            for layer_index in range(num_layers - 1):
                # 获取当前层到下一层的参数矩阵
                layer_theta = thetas[layer_index]  # 形状：25×785, 10×26
                # 计算下一层的净输入（加权和）
                layer_input = np.dot(
                    layer_theta, layers_activation
                )  # 第一次得到25×1, 第二次10×1
                # 计算下一层的激活值，并添加偏置项（值为1）
                layers_activation = np.vstack(
                    (np.array([[1]]), Sigmoid(layer_input).apply())
                )
                # 存储每层的净输入和激活值，便于反向传播时使用
                layers_inputs[layer_index + 1] = layer_input  # 存储净输入
                layers_activations[layer_index + 1] = layers_activation  # 存储激活值

            # 获取输出层的激活值（不包括偏置项）
            output_layer_activation = layers_activation[1:, :]

            # 计算每层的误差项（delta）
            delta = {}
            # 将标签转换为one-hot编码
            bitwise_label = np.zeros((num_label_types, 1))
            # 将当前样本的真实标签对应位置设为1
            bitwise_label[labels[example_index][0]] = 1

            # 计算输出层的误差项（预测值 - 真实值）
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # 反向传播误差，从倒数第二层到第一层
            # 遍历循环 L L-1 L-2 ...2，计算每层的误差项
            for layer_index in range(num_layers - 2, 0, -1):
                # 获取当前层到下一层的参数矩阵
                layer_theta = thetas[layer_index]
                # 获取下一层的误差项
                next_delta = delta[layer_index + 1]
                # 获取当前层的净输入
                layer_input = layers_inputs[layer_index]
                # 为净输入添加偏置项
                layer_input = np.vstack((np.array((1)), layer_input))

                # 计算当前层的误差项，使用反向传播公式：
                # delta^(l) = (theta^(l))^T * delta^(l+1) .* g'(z^(l))
                # 其中 g'(z^(l)) 是激活函数的梯度，.* 表示元素乘法
                delta[layer_index] = (
                    np.dot(layer_theta.T, next_delta) * Sigmoid(layer_input).apply()
                )

                # 去掉偏置项对应的误差项（不需要更新偏置项的梯度）
                delta[layer_index] = delta[layer_index][1:, :]

            # 计算每层参数的梯度增量，使用公式：
            # D^(l) = delta^(l+1) * (a^(l))^T
            for layer_index in range(num_layers - 1):
                # 计算梯度增量
                layer_delta = np.dot(
                    delta[layer_index + 1], layers_activations[layer_index].T
                )
                # 累加到总梯度中
                deltas[layer_index] = deltas[layer_index] + layer_delta

        # 对梯度进行归一化，除以样本数
        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)

        return deltas

    @staticmethod
    def cost_function(data, labels, thetas, layers):
        """
        计算神经网络的代价函数值（交叉熵损失）

        参数:
        data - 特征数据
        labels - 标签数据
        thetas - 当前的参数字典
        layers - 网络结构

        返回:
        cost - 代价函数值
        """
        # num_layers = len(layers)  # 网络层数
        num_examples = data.shape[0]  # 样本数
        num_labels = layers[-1]  # 输出类别数（10）

        # 前向传播，计算预测结果
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)

        # 将标签转换为one-hot编码矩阵
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            # 将每个样本的真实标签对应位置设为1
            bitwise_labels[example_index][labels[example_index][0]] = 1

        # 计算交叉熵损失，分为两部分：
        # 1. 对于真实类别为1的部分，计算 -log(h(x))
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        # 2. 对于真实类别为0的部分，计算 -log(1-h(x))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))

        # 合并两部分，并除以样本数进行归一化
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        """
        前向传播算法，计算神经网络的输出

        参数:
        data - 特征数据，形状为 (样本数, 特征数)
        thetas - 参数字典
        layers - 网络结构

        返回:
        输出层的激活值，形状为 (样本数, 输出类别数)
        """
        num_layers = len(layers)  # 网络层数
        num_examples = data.shape[0]  # 样本数
        # 初始输入层激活值就是特征数据
        in_layer_activation = data

        # 逐层前向传播
        for layer_index in range(num_layers - 1):
            # 获取当前层的参数矩阵
            theta = thetas[layer_index]
            # 计算下一层的激活值：sigmoid(X * theta^T)
            out_layer_activation = Sigmoid(np.dot(in_layer_activation, theta.T)).apply()
            # 添加偏置项（值为1）
            # 正常计算完之后是num_examples*25,但是要考虑偏置项 变成num_examples*26
            out_layer_activation = np.hstack(
                (np.ones((num_examples, 1)), out_layer_activation)
            )
            # 将当前层的输出作为下一层的输入
            in_layer_activation = out_layer_activation

        # 返回输出层结果，去掉偏置项（第一列）
        return in_layer_activation[:, 1:]

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """
        将一维参数数组重新转换为参数矩阵字典

        参数:
        unrolled_thetas - 展开的一维参数数组
        layers - 网络结构

        返回:
        thetas - 参数矩阵字典
        """
        num_layers = len(layers)  # 网络层数
        thetas = {}  # 创建参数字典
        unrolled_shift = 0  # 当前在一维数组中的位置

        # 逐层转换参数
        for layer_index in range(num_layers - 1):
            # 当前层和下一层的神经元数量
            in_count = layers[layer_index]  # 当前层神经元数
            out_count = layers[layer_index + 1]  # 下一层神经元数

            # 计算参数矩阵的尺寸
            thetas_width = in_count + 1  # 列数（包括偏置）
            thetas_height = out_count  # 行数
            thetas_volume = thetas_width * thetas_height  # 参数总数

            # 从一维数组中提取当前层的参数
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]

            # 重塑为矩阵形式，并存入字典
            thetas[layer_index] = layer_theta_unrolled.reshape(
                (thetas_height, thetas_width)
            )

            # 更新位置指针
            unrolled_shift = unrolled_shift + thetas_volume

        return thetas
