import os
import re

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 读取数据
df = pd.read_csv('processed_data.csv')

# 假设 'input_data' 和 'output_data' 分别对应于 DataFrame 的前几列和最后一列
# 将输入数据和输出数据转换为 NumPy 数组
X = df.iloc[:, :-1].values  # 所有行，除了最后一列
y = df.iloc[:, -1].values    # 所有行，最后一列
# 转换为NumPy数组
# X = np.array(input_data)
# y = np.array(output_data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
loss_values = []  # 用于存储损失值
# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 标准化输出数据
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # reshape以适应标准化
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)



# 构建 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 32)  # 第二个隐藏层
        self.fc3 = nn.Linear(32, 1)  # 输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型
model = MLP(X_train.shape[1])

# 定义模型保存的路径
model_dir = 'param'

# 查找最大数字的模型文件
def find_latest_model(model_dir):
    files = os.listdir(model_dir)
    model_files = [f for f in files if f.endswith('.pth')]  # 过滤出.pth文件

    max_epoch = -1
    latest_model = None

    for model_file in model_files:
        # 提取文件名中的数字
        match = re.search(r'_(\d+)\.pth', model_file)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model = model_file

    return latest_model

# 查找最新模型
latest_model_file = find_latest_model(model_dir)

# 尝试加载模型参数
if latest_model_file:
    model.load_state_dict(torch.load(os.path.join(model_dir, latest_model_file)))
    print(f"成功加载模型参数：{latest_model_file}")
else:
    print("未找到可加载的模型参数，使用新模型进行训练。")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100000

# 打开文件以写入损失值
with open('loss_values.txt', 'w') as f:
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # 清零梯度
        outputs = model(X_train_tensor)  # 前向传播
        loss = criterion(outputs.view(-1), y_train_tensor)   # 计算损失
        # 记录损失值

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        loss_values.append(loss.item())
        # 将当前损失值写入文件
        f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}\n')
        # 每100代保存一次模型参数
        if (epoch + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'param\\model_epoch_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()


# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.view(-1), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')



# 反标准化预测结果
predictions = scaler_y.inverse_transform(test_outputs.view(-1, 1)).flatten()
# 进行预测
# predictions = test_outputs.numpy()
print(predictions)
print(y_test)