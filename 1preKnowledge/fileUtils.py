import os
import pandas as pd
import torch

# file = open('../data/testFile.txt', 'w')
# try:
#     file.write("hello machineLeaning!\n")
# finally:
#     file.close()
#
# # with as 代替try finally
# with open('../data/testFile.txt', 'a') as file:
#     file.write('hello world!')


# 数据预处理
# 1.读取
data_file = os.path.join('..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data)
print(type(data))

# 2.处理缺失值
# inputs, outputs = data[:, 0:2], data[:, 2]
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)

# 平均值替代NaN
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 3.转化为张量
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)
