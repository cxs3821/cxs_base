import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 加载原始数据
df_train = pd.read_csv("/heartbeat/train.csv")
df_testA = pd.read_csv("/heartbeat/testA.csv")
# 查看训练和测试数据的前五条
print(df_train.head())
print('\n')
print(df_testA.head())
# 检查数据是否有NAN数据
print(df_train.isna().sum(),df_testA.isna().sum())
# 确认标签的类别及数量
print(df_train['label'].value_counts())
# 查看训练数据集特征
print(df_train.describe())
# 查看数据集信息
print(df_train.info())
# 绘制每种类别的折线图
ids = []
for id, row in df_train.groupby('label').apply(lambda x: x.iloc[2]).iterrows():
    ids.append(int(id))
    signals = list(map(float, row['heartbeat_signals'].split(',')))
    sns.lineplot(data=signals)

plt.legend(ids)
plt.show()


