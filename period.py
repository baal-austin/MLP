import pandas as pd

# 读取处理后的数据
df = pd.read_csv('input_data_season.csv')

# 确保日期时间列为 datetime 类型
df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['moment'])  # 根据你的列名进行调整

# 提取小时信息
df['Hour'] = df['Datetime'].dt.hour  # 提取小时

# 创建一个新的列，根据小时分组
def assign_period(hour):
    if 0 <= hour < 6:
        return 1
    elif 6 <= hour < 12:
        return 2
    elif 12 <= hour < 18:
        return 3
    elif 18 <= hour < 24:
        return 4

df['Period'] = df['Hour'].apply(assign_period)

# 根据Period进行分组并保存到不同的DataFrame中
period_1 = df[df['Period'] == 1]
period_2 = df[df['Period'] == 2]
period_3 = df[df['Period'] == 3]
period_4 = df[df['Period'] == 4]


# 输出每个数据集的头部信息以确认
print("Period 1 DataFrame:")
print(period_1.head())
print("\nPeriod 2 DataFrame:")
print(period_2.head())
print("\nPeriod 3 DataFrame:")
print(period_3.head())
print("\nPeriod 4 DataFrame:")
print(period_4.head())

df.to_csv('input_data_season.csv', index=False)