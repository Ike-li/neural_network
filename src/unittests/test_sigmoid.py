import unittest
import numpy as np

from src.sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def test_sigmoid_scalar(self):
        """测试sigmoid函数对标量值的处理"""
        # 测试0值
        sigmoid = Sigmoid(0)
        self.assertAlmostEqual(sigmoid.apply(), 0.5)
        # 测试正数
        sigmoid = Sigmoid(2)
        self.assertAlmostEqual(sigmoid.apply(), 0.8807970779778823)
        # 测试负数
        sigmoid = Sigmoid(-2)
        self.assertAlmostEqual(sigmoid.apply(), 0.11920292202211755)

    def test_sigmoid_array(self):
        """测试sigmoid函数对数组的处理"""
        # 测试一维数组
        input_array = np.array([0, 1, -1])
        sigmoid = Sigmoid(input_array)
        expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951])
        np.testing.assert_almost_equal(sigmoid.apply(), expected)

    def test_sigmoid_matrix(self):
        """测试sigmoid函数对矩阵的处理"""
        # 测试二维矩阵
        input_matrix = np.array([[1, 2], [3, 4]])
        sigmoid = Sigmoid(input_matrix)
        expected = np.array(
            [
                [0.7310585786300049, 0.8807970779778823],
                [0.9525741268224334, 0.9820137900379085],
            ]
        )
        np.testing.assert_almost_equal(sigmoid.apply(), expected)

    def test_sigmoid_large_values(self):
        """测试sigmoid函数对大值的处理（接近1）"""
        sigmoid = Sigmoid(10)
        self.assertAlmostEqual(sigmoid.apply(), 0.9999546021312976)

    def test_sigmoid_small_values(self):
        """测试sigmoid函数对小值的处理（接近0）"""
        sigmoid = Sigmoid(-10)
        self.assertAlmostEqual(sigmoid.apply(), 0.00004539786870243439)

    def test_gradient_scalar(self):
        """测试sigmoid梯度函数对标量值的处理"""
        # 输入0时的梯度，应该是0.25 (0.5 * (1 - 0.5))
        sigmoid = Sigmoid(0)
        self.assertAlmostEqual(sigmoid.gradient(), 0.25)

        # 测试正数
        sigmoid = Sigmoid(2)
        expected = 0.8807970779778823 * (1 - 0.8807970779778823)
        self.assertAlmostEqual(sigmoid.gradient(), expected)

        # 测试负数
        sigmoid = Sigmoid(-2)
        expected = 0.11920292202211755 * (1 - 0.11920292202211755)
        self.assertAlmostEqual(sigmoid.gradient(), expected)

    def test_gradient_array(self):
        """测试sigmoid梯度函数对数组的处理"""
        # 测试一维数组
        input_array = np.array([0, 1, -1])
        sigmoid = Sigmoid(input_array)

        # 手动计算期望结果
        s_values = 1 / (1 + np.exp(-input_array))
        expected = s_values * (1 - s_values)

        np.testing.assert_almost_equal(sigmoid.gradient(), expected)

    def test_gradient_matrix(self):
        """测试sigmoid梯度函数对矩阵的处理"""
        # 测试二维矩阵
        input_matrix = np.array([[1, 2], [3, 4]])
        sigmoid = Sigmoid(input_matrix)

        # 手动计算期望结果
        s_values = 1 / (1 + np.exp(-input_matrix))
        expected = s_values * (1 - s_values)

        np.testing.assert_almost_equal(sigmoid.gradient(), expected)


# 直接执行测试，确保if __name__ == "__main__"代码块被覆盖
unittest.main(argv=["first-arg-is-ignored"], exit=False)
