import pandas as pd
import xml.etree.ElementTree as ET
import os

# 将xml数据更改为csv数据，并替换指标名称
# 文件夹路径
folder_path = 'data'

# 用于存储所有数据的列表
all_data = []

# 遍历文件夹中的所有XML文件
for month in range(1, 13):
    file_name = f'C018_2023_{month}.xml'
    file_path = os.path.join(folder_path, file_name)

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 解析XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 提取数据
        for dia in root.findall('dia'):
            date = dia.attrib['Dia']
            for hora in dia.findall('hora'):
                moment = hora.attrib['Hora']
                meteoros = hora.find('Meteoros')

                # 创建一个字典来存储当前时刻的所有属性
                row = {'date': date, 'moment': moment}
                for child in meteoros:
                    # row[child.tag] = float(child.text)  # 将值转换为浮点数
                    value = child.text.strip() if child.text else None
                    # 检查值是否为None并转换
                    row[child.tag] = float(value) if value is not None else None
                all_data.append(row)

# 转换为DataFrame
df = pd.DataFrame(all_data)

# 存储为CSV文件
# df.to_csv('input_data.csv', index=False)
#
# 打印结果
# print("数据已成功存储为 input_data.csv")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Read the CSV file
# df = pd.read_csv('input_data.csv')

# View the original column names
# print("Original column names:")
# print(df.columns)


def renaming(df):  # Create a dictionary for renaming columns
    new_column_names = {
        'Dir.Med._a_900cm': 'AverageWindDirection',
        'Humedad._a_150cm': 'Humidity',
        'Irradia.._a_200cm': 'Irradiance',
        'Sig.Dir._a_900cm': 'SignalWindDirection',
        'Sig.Vel._a_900cm': 'SignalWindSpeed',
        'Tem.Aire._a_150cm': 'AirTemperature',
        'Vel.Max._a_900cm': 'MaxWindSpeed',
        'Vel.Med._a_900cm': 'AverageWindSpeed',
        # Add more attributes as needed
    }

    # Rename the columns
    df.rename(columns=new_column_names, inplace=True)

    # View the renamed column names
    print("Renamed column names:")
    print(df.columns)

    # Save the renamed DataFrame to a new CSV file
    df.to_csv('output_data.csv', index=False)


# Read the CSV file
# df = pd.read_csv('input_data.csv')

# Set the figure size
plt.figure(figsize=(12, 8))

# Draw box plots for each numerical column
sns.boxplot(data=df, orient='h')  # Horizontal orientation for better visibility

# Set title and labels
plt.title('Box Plots of All Attributes', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Attributes', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()