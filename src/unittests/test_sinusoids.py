import numpy as np

from sinusoids import Sinusoids


def test_sinusoids():
    sinusoids = Sinusoids()
    assert sinusoids


def test_sinusoids_generation():
    """测试正弦特征生成"""
    dataset = np.array([[0], [np.pi / 2], [np.pi]])
    sinusoid_degree = 3
    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)

    # 验证生成的正弦特征的形状
    expected_shape = (3, 3)
    assert generated_sinusoids.shape == expected_shape

    # 验证生成的正弦特征的值 - 修正期望值以匹配实际计算结果
    expected_values = np.array(
        [
            [0, 0, 0],
            [1, 0, -1],  # sin(π/2), sin(2*π/2), sin(3*π/2)
            [0, 0, 0],  # sin(π), sin(2*π), sin(3*π)
        ]
    )
    np.testing.assert_array_almost_equal(generated_sinusoids, expected_values)


def test_sinusoids_with_empty_array():
    """测试空数组输入"""
    dataset = np.array([[]])  # 改为二维空数组
    sinusoid_degree = 2

    # 空数组应该返回空矩阵而不是引发异常
    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)
    assert generated_sinusoids.shape == (1, 0)  # 一个样本，零特征


def test_sinusoids_with_zero_degree():
    """测试度数为0的情况"""
    dataset = np.array([[1], [2], [3]])
    sinusoid_degree = 0

    # 当度数为0时,应当返回空特征矩阵
    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)
    expected_shape = (3, 0)
    assert generated_sinusoids.shape == expected_shape


def test_sinusoids_with_2d_input():
    """测试二维输入数据"""
    dataset = np.array([[0, np.pi / 4], [np.pi / 2, np.pi / 3], [np.pi, np.pi / 6]])
    sinusoid_degree = 2

    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)

    # 验证输出形状: (样本数, 特征数*sinusoid_degree)
    expected_shape = (3, 4)  # 3个样本,每个样本2个特征,2个度数 -> 3 x (2*2)
    assert generated_sinusoids.shape == expected_shape

    # 验证第一个样本,第一个度数的结果 (sin(1*[0, pi/4]))
    expected_first_degree = np.sin(dataset)
    np.testing.assert_array_almost_equal(
        generated_sinusoids[:, 0:2], expected_first_degree
    )

    # 验证第一个样本,第二个度数的结果 (sin(2*[0, pi/4]))
    expected_second_degree = np.sin(2 * dataset)
    np.testing.assert_array_almost_equal(
        generated_sinusoids[:, 2:4], expected_second_degree
    )


def test_sinusoids_with_high_degree():
    """测试高次数正弦特征生成"""
    dataset = np.array([[0.1], [0.5], [1.0]])
    sinusoid_degree = 10

    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)

    # 验证输出形状
    expected_shape = (3, 10)  # 3个样本,1个特征,10个度数
    assert generated_sinusoids.shape == expected_shape

    # 验证最高次数的正弦特征
    expected_highest_degree = np.sin(10 * dataset)
    np.testing.assert_array_almost_equal(
        generated_sinusoids[:, 9:10], expected_highest_degree
    )


def test_sinusoids_with_negative_values():
    """测试负值输入"""
    dataset = np.array([[-1.0], [0], [1.0]])
    sinusoid_degree = 2

    generated_sinusoids = Sinusoids().generate(dataset, sinusoid_degree)

    # 验证输出形状
    expected_shape = (3, 2)
    assert generated_sinusoids.shape == expected_shape

    # 验证结果
    expected_values = np.array(
        [[np.sin(-1), np.sin(-2)], [0, 0], [np.sin(1), np.sin(2)]]
    )
    np.testing.assert_array_almost_equal(generated_sinusoids, expected_values)
