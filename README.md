# 神经网络实现项目

这个项目是一个神经网络库的实现，包含多层感知机（MLP）、特征处理和训练组件。该库主要用于教学目的，展示神经网络的基本原理和实现方法。

## 功能特点

- 基于NumPy的多层感知机实现
- 支持任意层数和神经元数量的网络结构
- 使用反向传播算法和梯度下降进行模型训练
- 提供特征处理工具：
  - 特征归一化
  - 多项式特征生成
  - 正弦特征变换
- 包含MNIST手写数字识别的示例应用

## MNIST手写数字识别优化

我们在MNIST手写数字识别任务上进行了一系列优化，将模型准确率从初始的约22%提高到了87.55%。主要优化措施包括：

1. **数据增强**：通过水平和垂直偏移等简单几何变换，将训练数据集扩展到原来的3倍，提高了模型鲁棒性和泛化能力。

2. **网络结构优化**：使用包含80个神经元的单隐藏层结构，在表示能力和避免过拟合之间取得平衡。

3. **参数初始化改进**：使用Xavier初始化方法代替简单随机初始化，减少梯度消失或爆炸的风险。

4. **L2正则化**：添加权重惩罚项，防止过拟合并提高模型泛化能力。

5. **学习率策略优化**：使用较大的初始学习率(0.3)快速接近最优解，然后使用学习率衰减策略（每200次迭代减半）在接近最优解时进行精细调整。

6. **延长训练时间**：增加迭代次数至800次，给模型更充分的学习时间。

通过以上优化，我们实现了训练集88.09%和测试集87.55%的准确率，且两者接近的差距表明模型具有良好的泛化能力。

## 项目结构

```
src/
├── __init__.py                 # 包初始化文件
├── multilayer_perceptron.py    # 多层感知机实现
├── normalize.py                # 特征归一化工具
├── polynomials.py              # 多项式特征生成工具
├── sigmoid.py                  # Sigmoid激活函数
├── sinusoids.py                # 正弦特征变换工具
├── training.py                 # 训练相关工具
├── mnist.py                    # MNIST数据集演示脚本（实现了优化的神经网络，准确率达87.55%）
├── pyproject.toml              # 项目配置文件
├── poetry.lock                 # 依赖锁定文件
├── data/                       # 数据集
│   ├── mnist-demo.csv          # MNIST数据集样本
│   ├── fashion-mnist-demo.csv  # Fashion-MNIST数据集样本
│   ├── iris.csv                # 鸢尾花数据集
│   ├── microchips-tests.csv    # 微芯片测试数据
│   ├── non-linear-regression-x-y.csv   # 非线性回归示例数据
│   ├── server-operational-params.csv   # 服务器操作参数数据
│   └── world-happiness-report-2017.csv # 世界幸福报告2017数据
└── unittests/                  # 单元测试
    ├── __init__.py
    ├── test_multilayer_perceptron.py
    ├── test_normalize.py
    ├── test_polynomials.py
    ├── test_sigmoid.py
    ├── test_sinusoids.py
    ├── test_training.py
    └── test_mnist.py           # MNIST测试
```

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/your-username/neural_network.git
cd neural_network
```

2. 使用Poetry安装依赖：

```bash
poetry install
```

## 使用方法

### 创建和训练模型

```python
import numpy as np
from multilayer_perceptron import MultilayerPerceptron

# 准备数据
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([[0], [1], [2], [0]])

# 定义网络结构：3个输入特征，4个隐藏神经元，3个输出类别
layers = [3, 4, 3]

# 创建模型
mlp = MultilayerPerceptron(X, y, layers, normalize_data=True)

# 训练模型
thetas, cost_history = mlp.train(max_iterations=1000, alpha=0.1)

# 预测
predictions = mlp.predict(X)
```

### 使用特征处理工具

```python
import numpy as np
from normalize import FeatureNormalizer
from polynomials import Polynomials
from sinusoids import Sinusoids

# 准备数据
data = np.array([[1, 2], [3, 4], [5, 6]])

# 特征归一化
normalizer = FeatureNormalizer()
normalized_data, mean, std = normalizer.normalize(data)

# 生成多项式特征
poly_features = Polynomials.generate(data, polynomial_degree=2, normalize_data=True)

# 生成正弦特征
sinusoid = Sinusoids()
sin_features = sinusoid.generate(data, sinusoid_degree=2)
```

### 运行MNIST演示

运行MNIST手写数字识别示例，展示我们优化后的神经网络模型（准确率87.55%）：

```bash
python src/mnist.py
```

此示例将展示：
- 手写数字样本可视化
- 训练过程中代价函数的变化
- 预测结果可视化（预测正确的样本为绿色，错误的为红色）
- 测试集准确率及错误示例分析

## 测试

使用pytest运行单元测试：

```bash
pytest
```

或者使用pre-commit来运行测试和代码质量检查：

```bash
pre-commit run --all-files
```

## 依赖

- NumPy：数值计算
- Pandas：数据处理
- Matplotlib：数据可视化
- pytest：单元测试

## 许可证

MIT
