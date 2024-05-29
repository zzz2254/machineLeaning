import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

"""
 定义方法来构造数据集
 Y=WX+b+c c是噪声 num_examples生成n个样本
"""
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 均值为0 方差为1的随机数，num_examples行，w列
    y = torch.matmul(X, w) + b   # matmul() 矩阵相乘
    y += torch.normal(0, 0.01, y.shape)
    # -1为自动计算，1为固定，即列向量为1
    return X, y.reshape((-1, 1))


"""
 定义真实的W和真实的b
"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2

"""
 构造数据集，数据为1000个
"""
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# 画出 features 和 labels 的散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

"""
 读取数据集
 使用随机的样本标号选取数据，每次返回batch_size个数据
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 样本个数 1000
    indices = list(range(num_examples))  # 下标 将1000个数字转为list [0, 1, 2, ..., 999]
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 为了随机访问样本，将1000个数字转为乱序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 测试
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

""" 
 初始化模型参数 
"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

"""
 定义模型
"""
def linreg(X, w, b):
    return torch.matmul(X, w) + b


""" 
 定义损失函数
 这里的平方损失函数为 (y_hat - y) ^2 /2 求和后乘 1/n
"""
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""
 定义优化算法 小批量-梯度下降算法
 lr学习率 batch_size批量大小
"""
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # 在这里求均值
            param.grad.zero_() # 梯度清零


"""
 定义超参数
"""
lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数 整个数据扫3遍
net = linreg
loss = squared_loss

"""
 训练
 2层for
 第一层for 将数据扫一遍
 第二层for 每一次拿出一个小批量x和y
 
"""
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    # 评价一下梯度下降的进度
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'训练的w为{w}')
print(f'训练的b为{b}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
