import numpy as np

from normalize import FeatureNormalizer
from polynomials import Polynomials
from sinusoids import Sinusoids


class Training:
    @staticmethod
    def prepare(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        准备数据集用于训练

        参数:
        data - 输入特征数据，形状为(样本数, 特征数)
        polynomial_degree - 多项式特征的最高次数，0表示不生成多项式特征
        sinusoid_degree - 正弦特征的最高次数，0表示不生成正弦特征
        normalize_data - 是否对数据进行归一化处理

        返回:
        data_processed - 处理后的特征数据，包括归一化、添加多项式/正弦特征和偏置项
        features_mean - 特征均值，用于后续数据的归一化
        features_deviation - 特征标准差，用于后续数据的归一化
        """

        # 计算样本总数
        num_examples = data.shape[0]

        # 复制原始数据，避免修改原始数据
        data_processed = np.copy(data)

        # 数据归一化处理
        features_mean = 0  # 特征均值，初始化为0
        features_deviation = 0  # 特征标准差，初始化为0
        data_normalized = data_processed  # 初始化归一化后的数据
        feature_normalize = FeatureNormalizer()  # 创建特征归一化器实例

        if normalize_data:  # 如果需要归一化
            # 对数据进行归一化处理，并获取均值和标准差
            (data_normalized, features_mean, features_deviation) = (
                feature_normalize.normalize(data_processed)
            )

            data_processed = data_normalized  # 更新处理后的数据为归一化后的数据

        # 生成正弦特征变换
        if sinusoid_degree > 0:  # 如果需要添加正弦特征
            # 生成正弦特征
            sinusoids = Sinusoids()
            sinusoids = sinusoids.generate(data_normalized, sinusoid_degree)
            # 将正弦特征与原特征连接
            data_processed = np.concatenate((data_processed, sinusoids), axis=1)

        # 生成多项式特征变换
        if polynomial_degree > 0:  # 如果需要添加多项式特征
            # 生成多项式特征
            polynomials = Polynomials()
            polynomials = polynomials.generate(
                data_normalized, polynomial_degree, normalize_data
            )
            # 将多项式特征与原特征连接
            data_processed = np.concatenate((data_processed, polynomials), axis=1)

        # 添加偏置特征（一列全为1的特征）
        data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

        return data_processed, features_mean, features_deviation
