import pandas as pd



# 读取处理后的数据
df = pd.read_csv('input_data_season.csv')

# 确保日期时间列为 datetime 类型
df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['moment'])

# 创建输入和输出数据
input_data = []
output_data = []

# 遍历数据，生成输入输出对
for i in range(0, len(df)):
    # if i % 144 == 0 and i > 0:  # 每144行后处理
    # 添加对应的时间段和季节
    it = []
    time_period = df.iloc[i]['Period']  # 时间段
    season_value = df.iloc[i]['Season']  # 季节值
    it.extend([time_period, season_value])
    it.extend(pd.concat([df.iloc[:i], df.iloc[len(df)-144+i:]])['Irradiance'].values)
    # input_data.append()
    # 取出前144行数据作为输入

    if i < 144:
        input_data[-1].append(it)
    else:
        input_data[-1].append(it)
    # 取出下一行的辐射值作为输出
    output_data.append(df.iloc[i]['Irradiance'])

# 将 input_data 转换为 DataFrame
input_df = pd.DataFrame(input_data)

# 将 output_data 转换为 DataFrame
output_df = pd.DataFrame(output_data, columns=['Target'])

# 合并输入和输出数据
final_df = pd.concat([input_df, output_df], axis=1)

# 存储为 CSV 文件
final_df.to_csv('processed_data.csv', index=False)

print("数据已成功保存为 processed_data.csv")


