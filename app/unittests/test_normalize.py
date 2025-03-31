class FeatureNormalizer:
    """对特征数据进行标准化处理

        采用z-score标准化方法：(x - μ) / σ
        其中 μ 是均值，σ 是标准差

        参数:
            features: 需要标准化的特征数据矩阵

        返回:
            features_normalized: 标准化后的特征数据
            features_mean: 特征的均值
            features_deviation: 特征的标准差
        """

    def __init__(self):
        pass

    def normalize(self):
        pass


def test_feature_normalize():
    feature_normalizer = FeatureNormalizer()
    assert feature_normalizer


def test_normalizer_normalize():
    feature_normalizer = FeatureNormalizer()
    feature_normalizer.normalize()




