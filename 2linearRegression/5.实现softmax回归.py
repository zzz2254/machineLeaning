import torch
from IPython import display
from d2l import torch as d2l

"""
 加载数据集
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
 初始化模型参数
"""
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))

"""
 定义softmax函数
"""
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X) #元素有正数和负数
# print(X_prob) # 每个元素都是正数，且每行总和为1
# print(X_prob.sum(dim=1, keepdim=True))


"""
 实现softmax模型
"""
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 根据索引取值
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y]) # 0.1 0.5

"""
 实现交叉熵损失函数
"""
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y))
