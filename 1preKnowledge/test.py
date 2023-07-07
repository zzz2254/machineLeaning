import torch

# 测试二阶求导
# x = torch.arange(4, dtype=torch.float32)
x = torch.tensor(4, dtype=torch.float32)
x.requires_grad_(True)
print(x)

y = x*x*x
# print(y)
# 求二阶导，在第一次求导保存了正向传播图
y.backward(retain_graph=True)
print(x.grad)
# 验证梯度
# print(x.grad == 3*x**2)
# 二阶求导
y.backward()
print(x.grad)


x.grad.zero_()
print(x.grad)
