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

# 构建 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # 在输出层应用 ReLU

# 数据处理函数
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, scaler, scaler_y

# 模型加载函数
def load_model(model, model_dir):
    def find_latest_model(model_dir):
        files = os.listdir(model_dir)
        model_files = [f for f in files if f.endswith('.pth')]

        max_epoch = -1
        latest_model = None

        for model_file in model_files:
            match = re.search(r'_(\d+)\.pth', model_file)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_model = model_file

        return latest_model

    latest_model_file = find_latest_model(model_dir)

    if latest_model_file:
        model.load_state_dict(torch.load(os.path.join(model_dir, latest_model_file)))
        print(f"成功加载模型参数：{latest_model_file}")
    else:
        print("未找到可加载的模型参数，使用新模型进行训练。")

# 训练模型函数
def train_model(model, X_train_tensor, y_train_tensor, num_epochs, model_dir):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_values = []

    with open('loss_values.txt', 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.view(-1), y_train_tensor)

            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            # f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}\n')

            if (epoch + 1) % 1000 == 0:
                loss_values.clear()  # 清空损失值列表

                torch.save(model.state_dict(), f'{model_dir}/model_epoch_{epoch + 1}.pth')
                print(f'Model saved at epoch {epoch + 1}')
            if (epoch + 1) % 10 == 0:
                f.write(f'{epoch + 1}, {loss.item():.6f}\n')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # return loss_values

# 绘制损失曲线函数
def plot_loss(loss_values):
    epochs = [item[0] for item in loss_values]
    losses = [item[1] for item in loss_values]

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

# 从文本文件读取损失值
def read_loss_values(file_path):
    with open(file_path, 'r') as f:
        loss_values = []
        for line in f:
            # 使用逗号分隔行，并去掉空格
            epoch_str, loss_str = line.strip().split(',')
            # 转换为 int 和 float
            epoch = int(epoch_str)
            loss = float(loss_str)  # 提取损失值并转换为 float
            loss_values.append((epoch, loss))  # 存储为元组
    return loss_values

# 评估模型函数
def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        # test_outputs = torch.clamp(test_outputs, min=0)  # 确保输出非负
        test_loss = nn.MSELoss()(test_outputs.view(-1), y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # 将 test_outputs 移动到 CPU，并转换为 NumPy 数组
        predictions = scaler_y.inverse_transform(test_outputs.cpu().view(-1, 1)).flatten()

        return predictions

# 主函数
# 主函数
def main():
    file_path = 'processed_data.csv'
    model_dir = 'param'
    num_epochs = 100000

    # 数据处理
    X_train, X_test, y_train, y_test, scaler, scaler_y = load_and_preprocess_data(file_path)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # 初始化模型
    model = MLP(X_train.shape[1]).to(device)

    # 加载模型
    load_model(model, model_dir)

    # 训练模型
    # train_model(model, X_train_tensor, y_train_tensor, num_epochs, model_dir)

    # 绘制损失曲线
    # loss_values = read_loss_values('loss_values.txt')
    # plot_loss(loss_values)

    # 评估模型
    predictions = evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y)

    y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    print(predictions[:10])
    print(y_test_inverse[:10])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()