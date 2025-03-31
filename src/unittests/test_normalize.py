from src.normalize import FeatureNormalizer
import numpy as np


def test_feature_normalize():
    feature_normalizer = FeatureNormalizer()
    assert feature_normalizer


def test_normalize_single_sample():
    """测试单个样本的情况"""
    features = np.array([[1, 2, 3]])
    normalized, mean, std = FeatureNormalizer.normalize(features)
    np.testing.assert_array_equal(normalized, features)  # 单个样本应该保持不变


def test_normalize_multiple_samples():
    """测试多个样本的情况"""
    features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    normalized, mean, std = FeatureNormalizer.normalize(features)

    # 验证均值计算正确
    expected_mean = np.array([4, 5, 6])
    np.testing.assert_array_almost_equal(mean, expected_mean)

    # 验证标准差计算正确
    expected_std = np.array([2.44948974, 2.44948974, 2.44948974])
    np.testing.assert_array_almost_equal(std, expected_std)

    # 验证标准化后的结果
    expected_normalized = np.array(
        [
            [-1.22474487, -1.22474487, -1.22474487],
            [0, 0, 0],
            [1.22474487, 1.22474487, 1.22474487],
        ]
    )
    np.testing.assert_array_almost_equal(normalized, expected_normalized)


def test_normalize_zero_std():
    """测试标准差为0的情况"""
    features = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    normalized, mean, std = FeatureNormalizer.normalize(features)

    # 验证标准差为0的特征被正确处理
    expected_normalized = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(normalized, expected_normalized)


def test_normalize_negative_values():
    """测试包含负值的情况"""
    features = np.array([[-1, -2, -3], [0, 0, 0], [1, 2, 3]])
    normalized, mean, std = FeatureNormalizer.normalize(features)

    # 验证标准化后的结果
    expected_normalized = np.array(
        [
            [-1.22474487, -1.22474487, -1.22474487],
            [0, 0, 0],
            [1.22474487, 1.22474487, 1.22474487],
        ]
    )
    np.testing.assert_array_almost_equal(normalized, expected_normalized)
