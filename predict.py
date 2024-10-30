import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from MLP2 import MLP, load_model, single_pre


# 假设您已经有训练好的模型和数据预处理的函数

def load_previous_day_data(date):
    # 将日期字符串转换为日期对象
    date = pd.to_datetime(date)
    date_str = date.strftime('%Y/%m/%d')  # 格式化为字符串
    # print(date_str)
    # 读取数据

    df = pd.read_csv('input_data_season.csv')  # 假设数据存储在这个文件中
    # 将 date_str 转换为日期并格式化
    date_obj = pd.to_datetime(date_str)
    date_str_formatted = f"{date_obj.year}/{date_obj.month}/{date_obj.day}"

    # 过滤出前一天的数据
    day_data = df[df['date'] == date_str_formatted]

    return day_data

# 定义季节判断函数
def determine_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 1  # 冬季
    elif month in [3, 4, 5]:
        return 2  # 春季
    elif month in [6, 7, 8]:
        return 3  # 夏季
    elif month in [9, 10, 11]:
        return 4  # 秋季

def determine_time_period(hour):
    if 0 <= hour < 6:
        return 1  # 00:00 - 06:00
    elif 6 <= hour < 12:
        return 2  # 06:00 - 12:00
    elif 12 <= hour < 18:
        return 3  # 12:00 - 18:00
    elif 18 <= hour < 24:
        return 4  # 18:00 - 24:00

date_str = "2023-4-11"
print(date_str)
date = pd.to_datetime(date_str)
season = determine_season(date)

previous_date = date - pd.Timedelta(days=1)  # 获取前一天的日期
pre_data = load_previous_day_data(previous_date) # 获取前一天数据
cur_data = load_previous_day_data(date)
irradiances = pre_data['Irradiance'].values
next_irra = None
# 创建时间段列表
# time_periods = []
# print(season)


predictions = []  # 存储预测值
true_values = []  # 存储真实值
moments = []  # 存储时刻
seasons = []  # 存储季节
time_periods = []  # 存储时间段
# 遍历每十分钟，计算属于哪个时间段
for i in range(144):
    # 计算小时和分钟
    hour = i // 6  # 每6个十分钟为一个小时
    minute = (i % 6) * 10  # 每个十分钟
    time_period = determine_time_period(hour)
    it = [season, time_period]
    if next_irra:
        irradiances = irradiances[1:]
        irradiances = np.append(irradiances,next_irra)
    it.extend(irradiances)
    it_array = np.array(it).reshape(1, -1)  # 转换为 1x2 数组
    # 构造时间字符串
    moment = f"{hour}:{minute:02d}"

    # 从 DataFrame 中获取对应的辐射值
    matching_row = cur_data[cur_data['moment'] == moment]
    irradiance_value = matching_row['Irradiance'].values  # 获取辐射值

    # 更新下一个辐射值
    next_irra = single_pre(it_array,irradiance_value)

    # 存储结果
    moments.append(moment)
    seasons.append(season)
    time_periods.append(time_period)
    predictions.append(next_irra)  # 预测辐射值
    true_values.append(irradiance_value[0] if irradiance_value.size > 0 else None)  # 真实辐射值

# 创建 DataFrame
results_df = pd.DataFrame({
    'Date': date_str,
    'Moment': moments,
    'Predicted Irradiance': predictions,
    'True Irradiance': true_values,
    'Season': seasons,
    'Time Period': time_periods
})

# 存储为 CSV 文件
results_df.to_csv(f'result\\{date_str}.csv', index=False)

print("预测与真实值已成功保存为 predictions_vs_true.csv")


# 输出时间段及其对应的时间
# for time, period in time_periods:
#     print(f"{time} 属于时间段: {period}")