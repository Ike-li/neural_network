import numpy as np


class Sigmoid:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self):
        """
        对输入矩阵应用sigmoid激活函数

        Sigmoid函数公式: f(x) = 1 / (1 + e^(-x))

        参数:
        matrix - 输入的NumPy数组或矩阵，可以是任意形状

        返回:
        相同形状的NumPy数组，每个元素都经过sigmoid函数变换
        sigmoid函数将输入映射到(0,1)区间，常用于神经网络的激活函数
        """

        return 1 / (1 + np.exp(-self.matrix))  # 应用sigmoid函数公式进行计算

    def gradient(self):
        """
        计算sigmoid函数在给定点处的梯度（导数）

        sigmoid函数的导数公式: f'(x) = f(x) * (1 - f(x))
        其中f(x)是sigmoid函数

        参数:
        matrix - 输入的NumPy数组或矩阵，表示需要计算梯度的点

        返回:
        相同形状的NumPy数组，包含每个位置处sigmoid函数的梯度值

        在神经网络的反向传播算法中，这个函数用于计算误差梯度
        """

        return self.apply() * (1 - self.apply())  # 应用sigmoid导数公式计算梯度
