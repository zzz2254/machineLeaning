import os
import pandas as pd
import torch


# 创建一个人工数据集
def make_csv():
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


make_csv()

# 删除缺失值最多的列
def drop_col(m):
    num = m.isna().sum()  # 获得缺失值统计信息
    num_dict = num.to_dict()  # 转为字典
    max_key = max(num_dict, key=num_dict.get)  # 取字典中最大值的键
    del m[max_key]  # 删除缺失值最多的列
    return m


"""
1.读取数据集
"""
data_file = os.path.join('..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data)  # “NaN”项代表缺失值


"""
2.处理缺失值： 
    插值法 用一个替代值弥补缺失值
    删除法 直接忽略缺失值
"""
# data共3列，inputs为前2列， outputs为后1列
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)

# 插值法 同一列的均值替换“NaN”项，因为第一列的均值为3，因此NaN被改为3
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 删除法 删除缺失值最多的列 2种方法
# drop_col(data)
# data = data.drop(pd.isna(data).sum(axis=0).idxmax(), axis=1)
# print(data)

# Alley列只接受两种类型的类别值“Pave”和“NaN”,将此列转换为两列“Alley_Pave”和“Alley_nan”
# Alley_Pave”的值设置为1，“Alley_nan”的值设置为0
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

"""
3.转换为张量格式
"""
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)
