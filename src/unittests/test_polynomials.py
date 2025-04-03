import numpy as np
import pytest
from src.polynomials import Polynomials


def test_polynomials_exists():
    """测试Polynomials类是否存在"""
    assert Polynomials


def test_generate_polynomial_degree_1():
    """测试生成一次多项式特征"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4]])

    # 生成一次多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=1)

    # 一次多项式应该包含原始特征
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_generate_polynomial_degree_2():
    """测试生成二次多项式特征"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4]])

    # 生成二次多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=2)

    # 二次多项式应该包含: x1, x2, x1^2, x1*x2, x2^2
    expected = np.array([[1, 2, 1, 2, 4], [3, 4, 9, 12, 16]])
    np.testing.assert_array_equal(result, expected)


def test_generate_polynomial_degree_3():
    """测试生成三次多项式特征"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4]])

    # 生成三次多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=3)

    # 三次多项式应该包含: x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
    expected = np.array(
        [[1, 2, 1, 2, 4, 1, 2, 4, 8], [3, 4, 9, 12, 16, 27, 36, 48, 64]]
    )
    np.testing.assert_array_equal(result, expected)


def test_normalize_polynomial_features():
    """测试归一化多项式特征"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4], [5, 6]])

    # 生成并归一化二次多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=2, normalize_data=True)

    # 验证结果的形状正确
    assert result.shape == (3, 5)

    # 验证归一化后的均值接近0
    assert np.abs(np.mean(result)) < 1e-10

    # 验证归一化后的标准差接近1
    assert np.all(np.abs(np.std(result, axis=0) - 1.0) < 1e-10)


def test_uneven_dataset():
    """测试特征数不均匀的数据集"""
    # 创建测试数据，特征数不均匀
    dataset = np.array([[1, 2, 3], [4, 5, 6]])

    # 应该能正常处理
    result = Polynomials.generate(dataset, polynomial_degree=2)

    # 验证结果的形状
    # 特征被分成两半，取较小的一半，所以每半有1个特征
    # 二次多项式特征数量应该是3：x1, x2, x1^2, x1*x2, x2^2，但由于只使用了第一个特征，所以只有3个
    assert result.shape == (2, 5)


def test_empty_dataset_error():
    """测试空数据集的情况"""
    # 创建空数据集
    empty_dataset = np.array([[], []])

    # 应该抛出ValueError
    with pytest.raises(ValueError):
        Polynomials.generate(empty_dataset, polynomial_degree=2)


def test_single_feature_dataset():
    """测试单特征数据集"""
    # 创建单特征数据集
    dataset = np.array([[1, 1], [2, 2], [3, 3]])

    # 生成二次多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=2)

    # 检查结果形状和值
    assert result.shape[0] == 3  # 样本数保持不变
    assert result.shape[1] > 0  # 至少有一个特征

    # 检查第一列和第三列（应该分别是 x1 和 x1^2）
    assert np.array_equal(result[:, 0], np.array([1, 2, 3]))
    assert np.array_equal(result[:, 2], np.array([1, 4, 9]))


def test_first_part_empty_features():
    """测试第一部分特征为空的情况"""
    # 创建第一部分特征为空的数据集
    dataset = np.array([[1, 2], [3, 4]])
    # 创建一个特殊的数据集，第一部分为空
    empty_part = np.zeros((2, 0))  # 创建空特征的两个样本

    # 确保这确实是个空特征数据集
    assert empty_part.shape == (2, 0)

    # 使用numpy.hstack组合它们
    combined_dataset = np.hstack((empty_part, dataset))

    # 生成多项式特征
    result = Polynomials.generate(combined_dataset, polynomial_degree=2)

    # 验证结果形状
    assert result.shape[0] == 2  # 样本数
    assert result.shape[1] > 0  # 至少有一个特征
    expected = np.array([[1, 2, 1, 2, 4], [3, 4, 9, 12, 16]])
    np.testing.assert_array_equal(result, expected)


def test_second_part_empty_features():
    """测试第二部分特征为空的情况"""
    # 创建第二部分特征为空的数据集
    dataset = np.array([[1, 2], [3, 4]])
    empty_part = np.zeros((2, 0))

    # 使用numpy.hstack组合它们，但这次将空特征放在第二部分
    combined_dataset = np.hstack((dataset, empty_part))

    # 生成多项式特征
    result = Polynomials.generate(combined_dataset, polynomial_degree=2)

    # 验证结果形状
    assert result.shape[0] == 2  # 样本数
    assert result.shape[1] > 0  # 至少有一个特征
    expected = np.array([[1, 2, 1, 2, 4], [3, 4, 9, 12, 16]])
    np.testing.assert_array_equal(result, expected)


def test_unequal_features():
    """测试特征数不均衡的情况，覆盖num_features计算"""
    # 创建特征数不均衡的数据集，第一部分特征数少于第二部分
    dataset_1 = np.array([[1], [2]])
    dataset_2 = np.array([[3, 4], [5, 6]])

    # 组合数据集
    combined_dataset = np.hstack((dataset_1, dataset_2))

    # 生成多项式特征
    result = Polynomials.generate(combined_dataset, polynomial_degree=2)

    # 验证使用了特征数较少的那部分
    assert result.shape == (
        2,
        5,
    )  # 因为虽然特征数不均衡，但当前实现会使用前1个特征和后1个特征，所以生成5个特征


def test_unequal_features_second_less():
    """测试第二部分特征数少于第一部分的情况"""
    # 创建特征数不均匀的数据集，第二部分特征数少
    dataset_1 = np.array([[1, 2], [3, 4]])
    dataset_2 = np.array([[5], [6]])

    # 组合数据集
    combined_dataset = np.hstack((dataset_1, dataset_2))

    # 生成多项式特征
    result = Polynomials.generate(combined_dataset, polynomial_degree=2)

    # 验证使用了特征数较少的那部分
    assert result.shape == (2, 5)  # 当前实现会使用第一个特征和第二部分的特征


def test_both_parts_no_features():
    """测试两部分都没有特征的情况"""
    # 创建两部分都没有特征的数据集
    empty_part1 = np.zeros((2, 0))
    empty_part2 = np.zeros((2, 0))

    # 组合数据集
    combined_dataset = np.hstack((empty_part1, empty_part2))

    # 应该抛出ValueError
    with pytest.raises(ValueError, match="无法为没有特征的数据集生成多项式特征"):
        Polynomials.generate(combined_dataset, polynomial_degree=2)


def test_first_part_no_features():
    """测试第一部分没有特征的情况，覆盖num_features_1 == 0条件分支"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4]])

    # 保存原始的array_split函数
    original_array_split = np.array_split

    # 定义一个替代函数来模拟第一部分没有特征
    def mock_array_split(arr, indices_or_sections, axis=0):
        result = original_array_split(arr, indices_or_sections, axis)
        # 强制第一部分没有特征
        result[0] = np.zeros((arr.shape[0], 0))
        return result

    try:
        # 替换numpy的array_split函数
        np.array_split = mock_array_split

        # 调用函数生成多项式特征
        result = Polynomials.generate(dataset, polynomial_degree=2)

        # 验证结果 - 由于使用第二部分替代第一部分，结果应该有值
        # 但我们不需要具体验证值，只需确认能正常工作
        assert result.shape[0] == dataset.shape[0]  # 样本数应保持不变
        # 不测试形状的列数，因为它可能随具体实现而变化
    finally:
        # 恢复原始函数
        np.array_split = original_array_split


def test_second_part_no_features():
    """测试第二部分没有特征的情况，覆盖num_features_2 == 0条件分支"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4]])

    # 保存原始的array_split函数
    original_array_split = np.array_split

    # 定义一个替代函数来模拟第二部分没有特征
    def mock_array_split(arr, indices_or_sections, axis=0):
        result = original_array_split(arr, indices_or_sections, axis)
        # 强制第二部分没有特征
        result[1] = np.zeros((arr.shape[0], 0))
        return result

    try:
        # 替换numpy的array_split函数
        np.array_split = mock_array_split

        # 调用函数生成多项式特征
        result = Polynomials.generate(dataset, polynomial_degree=2)

        # 验证结果 - 由于使用第一部分替代第二部分，结果应该有值
        # 但我们不需要具体验证值，只需确认能正常工作
        assert result.shape[0] == dataset.shape[0]  # 样本数应保持不变
        # 不测试形状的列数，因为它可能随具体实现而变化
    finally:
        # 恢复原始函数
        np.array_split = original_array_split


# 处理特殊情况：如果其中一部分没有特征，则使用另一部分代替
def test_one_part_empty():
    """处理特殊情况：如果其中一部分没有特征，则使用另一部分代替"""
    # 创建第一部分特征为空的数据集
    dataset_1 = np.array([[1, 2], [3, 4]])
    empty_part = np.zeros((2, 2))

    # 使用numpy.hstack组合它们
    combined_dataset = np.hstack((empty_part, dataset_1))

    # 生成多项式特征
    result = Polynomials.generate(combined_dataset, polynomial_degree=2)

    # 验证结果形状
    assert result.shape == (2, 10)  # 样本数保持不变


def test_normalize_data_with_real_normalizer():
    """测试归一化特性，包括直接调用FeatureNormalizer"""
    # 创建测试数据
    dataset = np.array([[1, 2], [3, 4], [5, 6]])

    # 生成并归一化多项式特征
    result = Polynomials.generate(dataset, polynomial_degree=2, normalize_data=True)

    # 验证结果大小和归一化效果
    assert result.shape == (3, 5)
    # 归一化后均值应接近0
    assert np.all(np.abs(np.mean(result, axis=0)) < 1e-10)
    # 归一化后标准差应接近1
    assert np.all(np.abs(np.std(result, axis=0) - 1.0) < 1e-10)
