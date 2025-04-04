import numpy as np


class Sinusoids:
    @staticmethod
    def generate(dataset, sinusoid_degree):
        """
        为数据集生成正弦特征.

        通过对原始特征应用正弦函数，生成新的特征：sin(x), sin(2x), sin(3x), ...
        这种特征变换可以帮助模型学习周期性规律

        参数:
        dataset - 输入特征数据，形状为 (样本数, 特征数)
        sinusoid_degree - 正弦特征的最高次数，生成 sin(1*x) 到 sin(sinusoid_degree*x)

        返回:
        sinusoids - 生成的正弦特征矩阵，形状为 (样本数, 特征数*sinusoid_degree)
        """

        # 获取样本数量
        num_examples = dataset.shape[0]

        # 创建一个空矩阵，用于存储生成的正弦特征
        sinusoids = np.empty((num_examples, 0))

        # 生成不同次数的正弦特征
        for degree in range(1, sinusoid_degree + 1):
            # 计算当前次数的正弦特征: sin(degree * x)
            sinusoid_features = np.sin(degree * dataset)

            # 将生成的特征添加到结果矩阵中
            sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

        return sinusoids
