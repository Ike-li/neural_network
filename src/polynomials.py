import numpy as np

from normalize import FeatureNormalizer


class Polynomials:
    @staticmethod
    def generate(dataset, polynomial_degree, normalize_data=False):
        """
        为数据集生成多项式特征

        多项式特征变换方法：x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, 等
        通过这种方式可以使线性模型学习非线性规律

        参数:
        dataset - 输入特征数据，形状为 (样本数, 特征数)
        polynomial_degree - 多项式的最高次数
        normalize_data - 是否对生成的多项式特征进行归一化处理

        返回:
        polynomials - 生成的多项式特征矩阵
        """

        # 将数据集分成两半，便于后续生成交叉项
        features_split = np.array_split(dataset, 2, axis=1)  # 沿列方向分割
        dataset_1 = features_split[0]  # 第一半特征
        dataset_2 = features_split[1]  # 第二半特征

        # 获取数据集的维度信息
        (num_examples_1, num_features_1) = dataset_1.shape  # 第一半特征的样本数和特征数
        (num_examples_2, num_features_2) = dataset_2.shape  # 第二半特征的样本数和特征数

        # 检查两部分数据是否都有特征
        if num_features_1 == 0 and num_features_2 == 0:
            raise ValueError("无法为没有特征的数据集生成多项式特征")

        # 处理特殊情况：如果其中一部分没有特征，则使用另一部分代替
        if num_features_1 == 0:
            dataset_1 = dataset_2
        elif num_features_2 == 0:
            dataset_2 = dataset_1

        # 取两部分中特征数较少的那一个，确保特征数一致
        num_features = (
            num_features_1 if num_features_1 < num_features_2 else num_features_2
        )
        dataset_1 = dataset_1[:, :num_features]  # 截取相同数量的特征
        dataset_2 = dataset_2[:, :num_features]

        # 创建一个空矩阵用于存储生成的多项式特征
        polynomials = np.empty((num_examples_1, 0))

        # 生成多项式特征
        # 外层循环控制多项式的次数，从1到polynomial_degree
        for i in range(1, polynomial_degree + 1):
            # 内层循环生成交叉项，j表示dataset_2的次数
            for j in range(i + 1):
                # 计算多项式特征: (x1^(i-j)) * (x2^j)
                polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2**j)
                # 将生成的特征添加到结果矩阵中
                polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

        # 如果需要归一化，则对生成的多项式特征进行归一化处理
        if normalize_data:
            feature_normalize = FeatureNormalizer()  # 创建特征归一化器实例
            polynomials = feature_normalize.normalize(polynomials)[
                0
            ]  # 只取归一化后的特征，不需要均值和标准差

        return polynomials
