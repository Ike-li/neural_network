import numpy as np


class FeatureNormalizer:
    """对特征数据进行标准化处理

    采用z-score标准化方法：(x - μ) / σ
    其中 μ 是均值，σ 是标准差

    注意：
    - 对于单个样本，保持原始值不变
    - 对于标准差为0的特征，normalized值设为0
    """

    @staticmethod
    def normalize(features):
        # 创建一个和features一样的矩阵，用来存储标准化后的结果
        features_normalized = np.copy(features).astype(float)

        # 计算均值
        features_mean = np.mean(features, 0)

        # 计算标准差
        features_deviation = np.std(features, 0)

        # 标准化操作
        if features.shape[0] > 1:
            features_normalized -= features_mean

        # 防止除以0
        features_deviation[features_deviation == 0] = 1
        features_normalized /= features_deviation

        # 返回结果
        return features_normalized, features_mean, features_deviation
