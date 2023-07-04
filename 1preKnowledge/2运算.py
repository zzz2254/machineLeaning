import torch

"""
1.相同形状的张量按元素计算
"""
# 按元素运算 加减乘除和求幂
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(x**y)
# print(torch.exp(x)) 不知道这个函数是干啥的

# 在做张量的拼接操作时，axis/dim设定了哪个轴，那对应的轴在拼接之后张量数会发生变化
# 多个张量连结 X1为dim=0按行连接；X2为dim=1按列连接
# reshape()
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
X1 = torch.cat((X, Y), dim=0)
Y2 = torch.cat((X, Y), dim=1)
# print(X)
# print(Y)
# print(X1)
# print(Y2)

# 如果X和Y在该位置相等，为true
# print(X == Y)


"""
2.广播机制 不同形状的张量按元素计算
"""
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
# b = torch.tensor([[0, 1], [2, 3]]) # 2行2列的张量不能和1行3列的张量广播
# print(a)
# print(b)
# 将两个矩阵广播为一个更大的 3x2矩阵，再按元素相加
# print(a+b)

"""
3.索引和切片
    按行切片和按列切片
    单元素赋值和多元素赋值
"""
# 获取单行或单列
# X[1:9] 能取到多行 含头不含尾，超出最大行数也不报错

# print(X[-1])  #最后一行
# print(X[0])  # 第0行
# print(X[1])  # 第1行
# print(X[:, 0])  # 第0列 获取单列和获取多列输出方式不一样
# print(X[:, 1])  # 第1列

# 获取多行或多列
# print(X[0:2])  # 第0、1行
# print(X[1:3, :])  # 第1、2行
# print(X[:, 0:2])  # 第0、1列
# print(X[:, 1:3])  # 第1、2列


# 单个元素赋值
# X[1,2] = 999
# print(X)

# 多个元素赋值 第0、1行都赋值为12
# X[0:2, :] = 12
# print(X)


"""
4.节约内存 [:] 或 +=
"""
before = id(Y)
# print(before)
Y = Y + X  # 给Y重新分配内存
# print(id(Y))
# print(id(Y) == before)

Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))
# Z[:] = X+Y
# print('id(Z):', id(Z))

before = id(X)
# X[:] = X + Y
# X -= Y
# print(id(X) == before)


"""
5.转换对象
"""
# 将NumPy张量转换为torch张量
A = X.numpy()
B = torch.tensor(A)
# print(type(A), type(B))

# 将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
# print(a, a.item(), float(a), int(a))


"""
6.练习
"""

# print(X > Y)  # 如果X在该位置>Y，为true

x = torch.arange(60).reshape(5, 4, 3)  # 5个4行3列矩阵
y = torch.arange(60).reshape(5, 4, 3)
print(x)
# print(x.ndim) # 3维张量
# z = x + y
# print(z)

x = torch.cat((x, y), dim=0)
# y = torch.cat((x, y), dim=1)
print(x)
# print(y)

c = torch.arange(120).reshape(5, 4, 3, 2)  # 4个3行2列矩阵为一个轴
# print(c)
# print(c.ndim)