import torch

# 需要一个地方来存储梯度。 重要的是，我们不会在每次对一个参数求导时都分配新的内存
x = torch.arange(4.0)
x.requires_grad_(True)
# x = torch.arange(4.0, requires_grad=True)  # 和上面两行等价
print(x)
# print(x.grad)  # 默认值是None

# 定义函数关于列向量x
y = 2 * torch.dot(x, x)
# print(y)

# 调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
# print(x.grad)

# 验证这个梯度是否计算正确
# print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
# print(y)
y.backward()
# print(x.grad)

"""
非标量变量的反向传播
    函数y为标量时，能直接y.backward()
    函数y为向量时，y.sum().backward()
"""
x.grad.zero_()
y = x * x
# print(y)
y.sum().backward()
# print(x.grad)

# x.grad.zero_()
# y.backward() # y为向量会报错，因为grad只能计算标量
# print(y.grad)

"""
分离计算
"""
# 反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
# print(x.grad)
# print(x.grad == u)

# 求 y = x*x 的导数
x.grad.zero_()
y.sum().backward()
# print(x.grad == 2 * x)

"""
Python控制流的梯度计算
"""
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()

print(a.grad == d/a)
