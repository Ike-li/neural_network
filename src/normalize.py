import numpy as np


class FeatureNormalizer:
    """
    特征数据标准化处理类

    采用z-score标准化方法：(x - μ) / σ
    其中 μ 是均值，σ 是标准差

    标准化的目的是使不同量纲的特征具有可比性，加速梯度下降收敛

    注意事项：
    - 对于单个样本，保持原始值不变
    - 对于标准差为0的特征，归一化值设为0
    """

    @staticmethod
    def normalize(features):
        """
        对特征数据进行标准化处理

        参数:
        features - 输入特征数据，形状为 (样本数, 特征数)

        返回:
        features_normalized - 归一化后的特征数据
        features_mean - 特征均值
        features_deviation - 特征标准差
        """
        # 创建一个和features一样的矩阵，用来存储标准化后的结果
        # 使用astype(float)确保数据类型为浮点数，避免整数计算带来的精度问题
        features_normalized = np.copy(features).astype(float)

        # 计算均值，axis=0表示沿列方向计算，即计算每个特征的均值
        features_mean = np.mean(features, 0)

        # 计算标准差，axis=0表示沿列方向计算，即计算每个特征的标准差
        features_deviation = np.std(features, 0)

        # 标准化操作：减去均值
        # 只有在样本数大于1时才执行，避免单个样本的情况
        if features.shape[0] > 1:
            features_normalized -= features_mean

        # 防止除以0：如果某特征的标准差为0，将其设为1，避免除法错误
        # 标准差为0意味着该特征在所有样本中的值都相同
        features_deviation[features_deviation == 0] = 1

        # 除以标准差完成标准化
        features_normalized /= features_deviation

        # 返回结果元组：(归一化后的特征, 均值, 标准差)
        return features_normalized, features_mean, features_deviation
