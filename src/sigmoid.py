import numpy as np


class Sigmoid:
    @staticmethod
    def apply(matrix):
        """
        对输入矩阵应用sigmoid激活函数

        Sigmoid函数公式: f(x) = 1 / (1 + e^(-x))

        参数:
        matrix - 输入的NumPy数组或矩阵，可以是任意形状

        返回:
        相同形状的NumPy数组，每个元素都经过sigmoid函数变换
        sigmoid函数将输入映射到(0,1)区间，常用于神经网络的激活函数
        """

        return 1 / (1 + np.exp(-matrix))  # 应用sigmoid函数公式进行计算
