# %matplotlib inline
import torch
from torch.distributions import multinomial
from d2l import torch as d2l
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6

# 真实的掷骰子，1~6每个数字出现的概率为1/6，约为0.1667
print(fair_probs)

"""
为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。 输出是另一个相同长度的向量：它在索引i处的值是采样结果中i出现的次数
    投掷1次骰子，1出现的下标，就是当前模拟中投出的结果 tensor([0., 0., 1., 0., 0., 0.]) 结果为3出现了1次
    投掷10次骰子，数字出现的下标，就是当前模拟中投出的结果
"""
print(multinomial.Multinomial(1, fair_probs).sample())
print(multinomial.Multinomial(10, fair_probs).sample())

# 模拟1000次投掷
count = multinomial.Multinomial(1000, fair_probs).sample()
print(count / 1000)

# 进行500组实验，每组抽取10个样本
count = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = count.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()

plt.show()
