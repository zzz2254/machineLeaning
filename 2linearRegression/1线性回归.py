# %matplotlib inline
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    """启动计时器"""

    def start(self):
        self.tik = time.time()

    """停止计时器并将时间记录在列表中"""

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    """返回平均时间"""

    def avg(self):
        return sum(self.times) / len(self.times)

    """返回时间总和"""

    def sum(self):
        return sum(self.times)

    """返回累计时间"""

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


c = torch.zeros([n])
timer = Timer()

for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

# 对比for和重载+的速度
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')


# 定义正态分布概率密度函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)
