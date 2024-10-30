import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 展示辐射值与其他值之间的相关性
# 读取CSV文件
df = pd.read_csv('input_data_processing.csv')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 查看数据基本信息
print(df.head())

# 计算相关性矩阵
correlation_matrix = df.corr()

# 提取与Irradiance相关的因素
# irradiance_correlation = correlation_matrix['Irradiance'].sort_values(ascending=False)

# 打印与Irradiance相关的因素
print("与Irradiance的相关性：")
# print(irradiance_correlation)

# 可视化相关性热图
# plt.figure(figsize=(12, 10))  # 增大图形尺寸以容纳更多指标
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='YlGnBu', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 10},
            linewidths=0.2, linecolor='black', )  # 设置线宽和线条颜色
# ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
#                  linewidths=0.2, cmap="YlGnBu",annot=True)

# 设置标题和标签
plt.title('Correlation heat map', fontsize=16)
plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签
plt.yticks(rotation=0)  # y 轴标签保持水平

# 调整布局以防止重叠
plt.tight_layout()
plt.show()
