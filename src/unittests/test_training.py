import numpy as np
from unittest.mock import patch

from normalize import FeatureNormalizer
from polynomials import Polynomials
from sinusoids import Sinusoids
from training import Training


def test_training_exists():
    """测试Training类是否存在"""
    assert Training


def test_prepare_basic():
    """测试基本的数据准备功能，不使用多项式或正弦特征"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 调用prepare方法
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True
    )

    # 验证结果
    assert data_processed.shape == (2, 3)  # 加了一列偏置项
    assert data_processed[:, 0].all() == 1  # 第一列应该全为1（偏置项）
    assert features_mean.shape == (2,)  # 应该有两个特征的均值
    assert features_deviation.shape == (2,)  # 应该有两个特征的标准差


def test_prepare_no_normalize():
    """测试不使用归一化的数据准备功能"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 调用prepare方法，不使用归一化
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=0, sinusoid_degree=0, normalize_data=False
    )

    # 验证结果
    assert data_processed.shape == (2, 3)  # 加了一列偏置项
    assert data_processed[:, 0].all() == 1  # 第一列应该全为1（偏置项）
    assert features_mean == 0  # 不归一化时均值应为0
    assert features_deviation == 0  # 不归一化时标准差应为0
    # 原始数据应该保持不变（除了添加的偏置项）
    np.testing.assert_array_equal(data_processed[:, 1:], data)


def test_prepare_with_polynomial():
    """测试使用多项式特征的数据准备功能"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 调用prepare方法，使用多项式特征
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=2, sinusoid_degree=0, normalize_data=True
    )

    # 验证结果
    # 数据形状应为：原始特征(2) + 多项式特征 + 偏置项(1)
    # 多项式特征数为：x1, x2, x1^2, x1*x2, x2^2，共5个
    assert data_processed.shape == (2, 8)  # 2(原始) + 5(多项式) + 1(偏置)
    assert data_processed[:, 0].all() == 1  # 第一列应该全为1（偏置项）


def test_prepare_with_sinusoid():
    """测试使用正弦特征的数据准备功能"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 调用prepare方法，使用正弦特征
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=0, sinusoid_degree=2, normalize_data=True
    )

    # 验证结果
    # 数据形状应为：原始特征(2) + 正弦特征(2*2=4) + 偏置项(1)
    assert data_processed.shape == (2, 7)  # 2(原始) + 4(正弦) + 1(偏置)
    assert data_processed[:, 0].all() == 1  # 第一列应该全为1（偏置项）


def test_prepare_with_both_features():
    """测试同时使用多项式和正弦特征的数据准备功能"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 调用prepare方法，同时使用多项式和正弦特征
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=2, sinusoid_degree=2, normalize_data=True
    )

    # 验证结果
    # 数据形状应为：原始特征(2) + 正弦特征(2*2=4) + 多项式特征(5) + 偏置项(1)
    assert data_processed.shape == (2, 12)  # 2(原始) + 4(正弦) + 5(多项式) + 1(偏置)
    assert data_processed[:, 0].all() == 1  # 第一列应该全为1（偏置项）


def test_prepare_empty_data():
    """测试处理空数据的情况"""
    # 创建空的测试数据（0个样本，2个特征）
    data = np.empty((0, 2))

    # 调用prepare方法
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=0, sinusoid_degree=0, normalize_data=False
    )

    # 验证结果
    assert data_processed.shape == (0, 3)  # 应该是0个样本，3个特征（包括偏置项）


def test_prepare_single_feature():
    """测试单特征数据的处理"""
    # 创建单特征测试数据
    data = np.array([[1], [2], [3]])

    # 调用prepare方法
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True
    )

    # 验证结果
    assert data_processed.shape == (3, 2)  # 3个样本，2个特征（包括偏置项）
    assert features_mean.shape == (1,)  # 应该有1个特征的均值
    assert features_deviation.shape == (1,)  # 应该有1个特征的标准差


@patch.object(FeatureNormalizer, "normalize")
@patch.object(Polynomials, "generate")
@patch.object(Sinusoids, "generate")
def test_integration_with_all_components(
    mock_sinusoids_generate, mock_polynomials_generate, mock_normalize
):
    """测试与所有组件的集成"""
    # 创建测试数据
    data = np.array([[1, 2], [3, 4]])

    # 设置模拟值
    mock_normalize.return_value = (data, np.array([2, 3]), np.array([1, 1]))
    mock_sinusoids_generate.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_polynomials_generate.return_value = np.array([[1.1, 1.2], [1.3, 1.4]])

    # 调用prepare方法，使用所有特征
    data_processed, features_mean, features_deviation = Training.prepare(
        data, polynomial_degree=2, sinusoid_degree=2, normalize_data=True
    )

    # 验证方法调用
    mock_normalize.assert_called_once()
    mock_sinusoids_generate.assert_called_once()
    mock_polynomials_generate.assert_called_once()

    # 验证结果
    assert data_processed.shape == (2, 7)  # 原始(2) + 正弦(2) + 多项式(2) + 偏置(1)
    np.testing.assert_array_equal(features_mean, np.array([2, 3]))
    np.testing.assert_array_equal(features_deviation, np.array([1, 1]))
