import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 从零实现线性回归

# 1.定义方法构造数据集
def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)


# 2.定义真实的w和b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 3.构造数据集
features, labels = synthetic_data(true_w, true_b, 1000)


# 4.读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 5.初始化模型参数
w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 6.定义模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b

# 7.定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 8.定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 9.定义超参数
lr = 0.04
num_epochs = 4
net = linreg
loss = squared_loss

# 10.训练
batch_size = 10
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')

print(f'训练的w为{w}')
print(f'训练的b为{b}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
