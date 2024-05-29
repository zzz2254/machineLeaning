import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 线性回归

# X = torch.normal(0, 1, (3, 4))
# print(X)
# true_w = torch.tensor([2, -3.4])
# print(true_w)

# features, labels = synthetic_data(true_w, true_b, 1000)
# print(features)
# num_examples = len(features)
# print(num_examples)
# indices = list(range(num_examples))
# print(indices)
# random.shuffle(indices)
# print(indices)

def synthetic_data(W, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(W)))  # 均值为0 方差为1的随机数，num_examples行，w列
    Y = torch.matmul(X, W) + b   # matmul() 矩阵相乘
    Y += torch.normal(0, 0.01, Y.shape)  # 噪声
    # -1为自动计算，1为固定，即列向量为1
    return X, Y.reshape((-1, 1))



true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 样本个数 1000
    indices = list(range(num_examples))  # 下标 将1000个数字转为list [0, 1, 2, ..., 999]
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 为了随机访问样本，将1000个数字转为乱序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
