import unittest
import numpy as np

from src.multilayer_perceptron import MultilayerPerceptron


class TestMultilayerPerceptron(unittest.TestCase):
    def setUp(self):
        """测试前的设置，创建简单的测试数据"""
        # 创建简单的特征数据，4个样本，3个特征
        self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        # 创建标签数据，4个样本，分别属于0、1、2、0类
        self.labels = np.array([[0], [1], [2], [0]])
        # 定义网络结构：3个输入特征，4个隐藏神经元，3个输出类别
        self.layers = [3, 4, 3]

    def test_init(self):
        """测试初始化方法"""
        # 测试默认初始化（不归一化数据）
        mlp = MultilayerPerceptron(self.data, self.labels, self.layers)
        self.assertEqual(mlp.data.shape[1], 4)  # 3特征 + 1偏置
        self.assertEqual(
            len(mlp.thetas), 2
        )  # 应该有2个theta矩阵（输入到隐藏，隐藏到输出）
        self.assertEqual(
            mlp.thetas[0].shape, (4, 4)
        )  # 第一层theta：4隐藏神经元 × (3输入 + 1偏置)
        self.assertEqual(
            mlp.thetas[1].shape, (3, 5)
        )  # 第二层theta：3输出神经元 × (4隐藏神经元 + 1偏置)

        # 测试带归一化的初始化
        mlp_norm = MultilayerPerceptron(
            self.data, self.labels, self.layers, normalize_data=True
        )
        self.assertEqual(mlp_norm.normalize_data, True)

    def test_thetas_init(self):
        """测试参数初始化方法"""
        thetas = MultilayerPerceptron.thetas_init(self.layers)
        self.assertEqual(len(thetas), 2)  # 应该有2个theta矩阵
        self.assertEqual(
            thetas[0].shape, (4, 4)
        )  # 第一层theta：4隐藏神经元 × (3输入 + 1偏置)
        self.assertEqual(
            thetas[1].shape, (3, 5)
        )  # 第二层theta：3输出神经元 × (4隐藏神经元 + 1偏置)
        # 确保初始化在合理范围内（0-0.05）
        self.assertTrue(np.all(thetas[0] >= 0) and np.all(thetas[0] <= 0.05))
        self.assertTrue(np.all(thetas[1] >= 0) and np.all(thetas[1] <= 0.05))

    def test_thetas_unroll(self):
        """测试参数展开方法"""
        # 创建已知的测试theta参数
        thetas = {
            0: np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            1: np.array([[9, 10, 11], [12, 13, 14]]),
        }
        # 展开参数
        unrolled = MultilayerPerceptron.thetas_unroll(thetas)
        # 计算期望的结果
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        # 比较结果
        np.testing.assert_array_equal(unrolled, expected)

    def test_thetas_roll(self):
        """测试参数重塑方法"""
        # 创建已知的一维参数
        unrolled = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        # 定义网络结构
        layers = [3, 2, 2]
        # 重塑参数
        thetas = MultilayerPerceptron.thetas_roll(unrolled, layers)
        # 检查结果
        self.assertEqual(len(thetas), 2)
        np.testing.assert_array_equal(thetas[0], np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        np.testing.assert_array_equal(thetas[1], np.array([[9, 10, 11], [12, 13, 14]]))

    def test_feedforward_propagation(self):
        """测试前向传播方法"""
        # 创建测试数据和参数，确保维度匹配
        # 3个特征，需要添加偏置项，因此创建包含偏置项的数据
        test_data = np.array([[1, 0.1, 0.2, 0.3], [1, 0.4, 0.5, 0.6]])
        layers = [3, 2, 2]
        thetas = {
            0: np.ones((2, 4)) * 0.1,  # 2个隐藏层神经元，每个连接3个输入+1个偏置
            1: np.ones((2, 3)) * 0.1,  # 2个输出神经元，每个连接2个隐藏层+1个偏置
        }

        # 计算前向传播结果
        output = MultilayerPerceptron.feedforward_propagation(test_data, thetas, layers)

        # 验证结果形状
        self.assertEqual(output.shape, (2, 2))  # 2个样本，2个输出类别

        # 手动计算第一个样本的结果
        # 第一层激活值 = sigmoid(X * theta^T)
        layer1_input = np.dot(test_data, thetas[0].T)
        layer1_activation = 1 / (1 + np.exp(-layer1_input))
        # 添加偏置项
        layer1_with_bias = np.hstack(
            (np.ones((test_data.shape[0], 1)), layer1_activation)
        )
        # 第二层激活值
        layer2_input = np.dot(layer1_with_bias, thetas[1].T)
        expected_output = 1 / (1 + np.exp(-layer2_input))

        # 验证输出近似正确
        np.testing.assert_almost_equal(output, expected_output, decimal=6)

    def test_cost_function(self):
        """测试代价函数方法"""
        # 创建包含偏置项的测试数据
        test_data = np.array([[1, 0.1, 0.2, 0.3], [1, 0.4, 0.5, 0.6]])
        test_labels = np.array([[0], [1]])
        layers = [3, 2, 2]

        # 创建特定的参数
        thetas = {0: np.ones((2, 4)) * 0.1, 1: np.ones((2, 3)) * 0.1}

        # 计算代价
        cost = MultilayerPerceptron.cost_function(
            test_data, test_labels, thetas, layers
        )

        # 验证代价为正数
        self.assertGreater(cost, 0)

    def test_back_propagation(self):
        """测试反向传播方法"""
        # 创建包含偏置项的测试数据
        test_data = np.array([[1, 0.1, 0.2, 0.3], [1, 0.4, 0.5, 0.6]])
        test_labels = np.array([[0], [1]])
        layers = [3, 2, 2]

        # 创建参数
        thetas = {0: np.ones((2, 4)) * 0.1, 1: np.ones((2, 3)) * 0.1}

        # 计算梯度
        deltas = MultilayerPerceptron.back_propagation(
            test_data, test_labels, thetas, layers
        )

        # 验证梯度形状
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[0].shape, (2, 4))
        self.assertEqual(deltas[1].shape, (2, 3))

    def test_gradient_step(self):
        """测试梯度步骤方法"""
        # 创建包含偏置项的测试数据
        test_data = np.array([[1, 0.1, 0.2, 0.3], [1, 0.4, 0.5, 0.6]])
        test_labels = np.array([[0], [1]])
        layers = [3, 2, 2]

        # 创建一维参数 (2*4 + 2*3 = 14个参数)
        unrolled_theta = np.ones(14) * 0.1

        # 计算梯度步骤
        gradient = MultilayerPerceptron.gradient_step(
            test_data, test_labels, unrolled_theta, layers
        )

        # 验证梯度形状
        self.assertEqual(gradient.shape, (14,))

    def test_gradient_descent(self):
        """测试梯度下降方法"""
        # 创建包含偏置项的测试数据
        test_data = np.array([[1, 0.1, 0.2, 0.3], [1, 0.4, 0.5, 0.6]])
        test_labels = np.array([[0], [1]])
        layers = [3, 2, 2]

        # 创建一维参数
        unrolled_theta = np.ones(14) * 0.1

        # 执行梯度下降，只迭代少量次数
        optimized_theta, cost_history = MultilayerPerceptron.gradient_descent(
            test_data, test_labels, unrolled_theta, layers, max_iterations=3, alpha=0.1
        )

        # 验证结果
        self.assertEqual(optimized_theta.shape, (14,))
        self.assertEqual(len(cost_history), 3)
        # 验证代价是否下降
        self.assertLessEqual(cost_history[-1], cost_history[0])

    def test_train(self):
        """测试训练方法"""
        # 初始化模型
        mlp = MultilayerPerceptron(self.data, self.labels, self.layers)

        # 训练模型，只迭代少量次数
        thetas, cost_history = mlp.train(max_iterations=3, alpha=0.1)

        # 验证结果
        self.assertEqual(len(thetas), 2)
        self.assertEqual(thetas[0].shape, (4, 4))
        self.assertEqual(thetas[1].shape, (3, 5))
        self.assertEqual(len(cost_history), 3)

    def test_predict(self):
        """测试预测方法"""
        # 初始化模型
        mlp = MultilayerPerceptron(self.data, self.labels, self.layers)

        # 训练模型，只迭代少量次数
        mlp.train(max_iterations=3, alpha=0.1)

        # 预测
        predictions = mlp.predict(self.data)

        # 验证预测结果
        self.assertEqual(predictions.shape, (4, 1))
        # 确认所有预测类别都在合法范围内
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions < 3))


# 直接执行测试，确保if __name__ == "__main__"代码块被覆盖
unittest.main(argv=["first-arg-is-ignored"], exit=False)
