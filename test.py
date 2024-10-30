import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 读取处理后的数据
df = pd.read_csv('input_data_season.csv')

# 确保日期时间列为 datetime 类型
df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['moment'])  # 根据你的列名进行调整

# 提取时刻和日期信息
df['Hour'] = df['Datetime'].dt.hour  # 提取小时
df['Date'] = df['Datetime'].dt.date  # 提取日期

# 计算每个季节每日每时刻的辐射值平均
seasonal_means = df.groupby(['Season', 'Hour'])['Irradiance'].mean().unstack(level=0)

# 对每个季节进行曲线拟合
plt.figure(figsize=(12, 8))

for season in seasonal_means.columns:
    # 提取小时和对应的辐射值
    x = seasonal_means.index.values
    y = seasonal_means[season].values

    # 进行插值拟合
    f = interp1d(x, y, kind='cubic')  # 使用三次插值
    x_new = np.linspace(0, 23, num=100)  # 生成更细的小时范围
    y_new = f(x_new)  # 计算拟合的y值

    # 绘制拟合曲线
    plt.plot(x_new, y_new, label=f'Season {season}')

# 设置每小时6个数据点（每10分钟一个点）
points_per_hour = 1
total_hours = 25

# 格式化横坐标为时间形式
plt.xticks(ticks=np.arange(0, total_hours * points_per_hour, points_per_hour * 3),
           labels=[f'{h:02d}:00' for h in range(0, total_hours, 3)])  # 显示00:00, 03:00形式

# 添加图例和标签
plt.title('Fitted Hourly Irradiance by Season', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Fitted Irradiance Value', fontsize=14)
plt.yticks(np.arange(0, seasonal_means.values.max() + 100, 100))  # Y轴刻度设置为100一个刻度
plt.grid()
plt.legend(title='Seasons', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # 在y=0的位置绘制水平线
plt.ylim(bottom=0)  # 设置y轴下限为0
plt.xlim(left=0)  # 设置y轴下限为0
# 显示图形
plt.tight_layout()
plt.show()
