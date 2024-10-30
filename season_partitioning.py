import numpy as np
import pandas as pd

# 读取数据
df = pd.read_csv('input_data_processing.csv')

df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['moment'])  # 请根据你的列名修改

# 定义一个函数来确定季节
def get_season(month):
    if month in [3, 4, 5]:  # 春季
        return 1
    elif month in [6, 7, 8]:  # 夏季
        return 2
    elif month in [9, 10, 11]:  # 秋季
        return 3
    else:  # 冬季
        return 4

# 添加一个新的列 'Season'，通过应用 get_season 函数
df['Season'] = df['Datetime'].dt.month.apply(get_season)


# 分割数据集
spring_data = df[df['Season'] == 1]
summer_data = df[df['Season'] == 2]
autumn_data = df[df['Season'] == 3]
winter_data = df[df['Season'] == 4]

# 打印结果（可选）
print("春季数据：")
print(spring_data.head())
print("夏季数据：")
print(summer_data.head())
print("秋季数据：")
print(autumn_data.head())
print("冬季数据：")
print(winter_data.head())

df.to_csv('input_data_season.csv', index=False)