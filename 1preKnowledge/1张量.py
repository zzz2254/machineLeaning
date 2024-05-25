import torch

"""
张量表示一个由数值组成的数组，这个数组可能有多个维度。
张量 对应数学的向量
具有两个轴的张量对应数学上的矩阵
"""
# 创建一个向量
x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())

# 指定行列 当限定了总数为12，给出行为3，列为-1，自动得出列4
x = x.reshape(3, 4)
x = x.reshape(-1, 4)
x = x.reshape(3, -1)
print(x)

# rehape 相当于数据库的view视图
e = torch.arange(12)
f = e.reshape((3, 4))
f[:] = 2
print(e)


# 创建全为0的张量 深度2 长3 宽4
y = torch.zeros(2, 3, 4)
# z = torch.zeros((2,3,4))
print(y)
print(y.shape)
# print(z)

# 创建全为1的张量
a = torch.ones(2, 3, 4)
# print(a)

# 创建一个张量，每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
b = torch.randn(3, 4)
# print(b)

# 给张量赋予确定的值
c = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(c)
