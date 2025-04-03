import unittest
import numpy as np

from src.sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def test_sigmoid_scalar(self):
        """测试sigmoid函数对标量值的处理"""
        # 测试0值
        self.assertAlmostEqual(Sigmoid.apply(0), 0.5)
        # 测试正数
        self.assertAlmostEqual(Sigmoid.apply(2), 0.8807970779778823)
        # 测试负数
        self.assertAlmostEqual(Sigmoid.apply(-2), 0.11920292202211755)

    def test_sigmoid_array(self):
        """测试sigmoid函数对数组的处理"""
        # 测试一维数组
        input_array = np.array([0, 1, -1])
        expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951])
        np.testing.assert_almost_equal(Sigmoid.apply(input_array), expected)

    def test_sigmoid_matrix(self):
        """测试sigmoid函数对矩阵的处理"""
        # 测试二维矩阵
        input_matrix = np.array([[1, 2], [3, 4]])
        expected = np.array(
            [
                [0.7310585786300049, 0.8807970779778823],
                [0.9525741268224334, 0.9820137900379085],
            ]
        )
        np.testing.assert_almost_equal(Sigmoid.apply(input_matrix), expected)

    def test_sigmoid_large_values(self):
        """测试sigmoid函数对大值的处理（接近1）"""
        self.assertAlmostEqual(Sigmoid.apply(10), 0.9999546021312976)

    def test_sigmoid_small_values(self):
        """测试sigmoid函数对小值的处理（接近0）"""
        self.assertAlmostEqual(Sigmoid.apply(-10), 0.00004539786870243439)


# 直接执行测试，确保if __name__ == "__main__"代码块被覆盖
unittest.main(argv=["first-arg-is-ignored"], exit=False)
