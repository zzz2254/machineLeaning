import torch

# 测试二阶求导
# # x = torch.arange(4, dtype=torch.float32)
# x = torch.tensor(4, dtype=torch.float32)
# x.requires_grad_(True)
# print(x)
#
# y = x*x*x
# # print(y)
# # 求二阶导，在第一次求导保存了正向传播图
# y.backward(retain_graph=True)
# print(x.grad)
# # 验证梯度
# # print(x.grad == 3*x**2)
# # 二阶求导
# y.backward()
# print(x.grad)
#
#
# x.grad.zero_()
# print(x.grad)

# 测试normal()
# 均值为0 方差为1的随机数，3行，2列
x = torch.normal(0, 1, (3, 2))
print(x)
y = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
print(y)
a = torch.zeros(1, requires_grad=True)
print(a)

# 求 y=x^3 的梯度
x = torch.arange(5.0, requires_grad=True)
y = x**3
y.sum().backward()
print(x)
print(x.grad)
# 默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
