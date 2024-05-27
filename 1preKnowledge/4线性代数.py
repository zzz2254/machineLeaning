import torch

"""
1.标量和向量
"""
# 标量 实例化两个标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)

# print(x+y, x*y, x/y, x**y)

"""
2.长度 维度 形状
    大量文献认为列向量是向量的默认方向，在本书中也是如此。
"""
x = torch.arange(4.)
# print(x)
# print(x[3])
# print(len(x))
# print(x.ndim)
# print(x.shape)

# x = torch.arange(8).reshape((2, 4))  # 创建2行4列的矩阵
# print(x)
# print(x.ndim)  # 轴数
# print(x.shape)

"""
3.矩阵
"""
A = torch.arange(20).reshape(5, 4)
# print(A)
# print(A.T)  # 矩阵转置

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])  # 对称矩阵
# print(B)
# print(B == B.T)  # 对称矩阵的转置和原矩阵相同


"""
4.张量
    标量是0阶张量，向量是一阶张量，矩阵是二阶张量
"""
X = torch.arange(24).reshape(2, 3, 4)
# print(X)

"""
5.张量算法的基本性质
    将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法
    按元素乘法称为Hadamard积
"""

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # clone()会重新开辟内存并赋值， B = A 只是给一个指针，还是同一个内存空间
# print(A)
# print(id(A), id(B))
# print(A+B)
# print(A*B) # 相同位置的按元素乘积
# print(torch.dot(A, B)) # 相同位置的按元素乘积的和

# 张量乘以或加上一个标量不会改变张量的形状
a = 2
X = torch.arange(24).reshape(2, 3, 4)
# print(X)
# print(a + X)
# print((a * X).shape)

"""
6.降维
    默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。
    我们还可以指定张量沿哪一个轴来通过求和降低维度。
"""
# 指定求和的轴
A = torch.arange(2*20, dtype=torch.float32).reshape(2, 5, 4)
# print(A)
# print(A.shape)
# print(A.sum())

# 对哪个维度求和，就去消去那一个维度
# reshape(2, 5, 4)
# 维度顺序是0  1  2

A_sum_axis0 = A.sum(axis=0) # 对第0个维度求和
# print(A_sum_axis0)
# print(A_sum_axis0.shape)
# print(A_sum_axis0.sum())

A_sum_axis1 = A.sum(axis=1) # 对第1个维度求和
# print(A_sum_axis1)
# print(A_sum_axis1.shape)
# print(A_sum_axis1.sum())

A_sum_axis2 = A.sum(axis=2) # 对第2个维度求和
# print(A_sum_axis2)
# print(A_sum_axis2.shape)
# print(A_sum_axis2.sum())

A_sum_axis01 = A.sum(axis=[0, 1]) # 两个维度求和
# print(A_sum_axis01)
# print(A_sum_axis01.shape)
# print(A_sum_axis01.sum())

# 平均值
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A)
# print(A.mean(dtype=torch.float32))
# print(A.sum()/A.numel())

# print(A.mean(axis=0))
# print(A.sum(axis=0)/A.shape[0])

# 非降维求和 计算求和或者均值时，保持轴数不变，这样就可以用广播机制
sum_A = A.sum(axis=1, keepdims=True) # torch.Size([5, 1]) 第一个维度的4变为1
sum_A_NK = A.sum(axis=1) # torch.Size([5]) 去掉第一个维度
# print(sum_A.shape)
# print(sum_A)
# print(A/sum_A)

# print(sum_A_NK)
# print(sum_A_NK.shape)

# 在某个轴计算 累计总和
# print(A.cumsum(axis=0))

"""
7.点积
    相同位置的按元素乘积的和
"""
x = torch.arange(4.)
y = torch.ones(4, dtype=torch.float32)
# print(x)
# print(y)
# print(x*y)
# print(torch.dot(x, y))
# print(torch.sum(x*y))

"""
8.矩阵-向量积
    相同位置的按元素乘积的和
"""
# print(A.shape)
# print(x.shape)
# print(A)
# print(x)
# print(torch.mv(A, x))  # 矩阵A和向量x 求矩阵向量积，相同位置相乘，再同行数字相加

"""
9.矩阵-矩阵乘法
    C = AB 两个矩阵，看作A的行向量乘B的列向量
"""
B = torch.ones(4, 3)
# print(A)
# print(B)
# print(torch.mm(A, B))

"""
10.范数
    向量的范数可以简单形象的理解为向量的长度
    在线性代数中，向量范数是将向量映射到标量的函数
    L2范数是向量元素平方和的平方根 ||x||2
    L1范数为向量元素的绝对值之和   ||x||1
    Lp范数是矩阵元素平方和的平方根，L2范数和L1范数是Lp范数的特例
"""
u = torch.tensor([3.0, -4.0])
print(torch.abs(u).sum())  # L1范数
v = torch.ones((4, 9))
print(v)
print(torch.norm(v)) # L2范数
