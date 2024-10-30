import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('input_data.csv')

# 定义一个函数来处理异常值
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].mask((data[column] < lower_bound) | (data[column] > upper_bound)).interpolate()
    return data

# 处理所有数值列的异常值
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    handle_outliers(df, col)

# 保存处理后的数据到CSV文件
df.to_csv('input_data_processing.csv', index=False)

# 绘制处理后的箱型图
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, orient='h')
plt.title('Box Plots of All Attributes After Outlier Handling', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Attributes', fontsize=14)
plt.tight_layout()
plt.show()
