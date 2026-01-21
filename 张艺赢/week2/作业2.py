import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
# 生成更多的数据点以获得更平滑的曲线
X_numpy = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
# 添加一些噪声使拟合更具挑战性
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(200, 1)

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print(f"X范围: [{X_numpy.min():.2f}, {X_numpy.max():.2f}]")
print(f"y范围: [{y_numpy.min():.2f}, {y_numpy.max():.2f}]")
print("---" * 10)

# 2. 构建多层神经网络模型
class MultiLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MultiLayerNet, self).__init__()
        # 第一层：输入层 -> 隐藏层1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # 第二层：隐藏层1 -> 隐藏层2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # 第三层：隐藏层2 -> 输出层
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # 第一层计算 + ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 第二层计算 + ReLU激活函数
        x = torch.relu(self.fc2(x))
        # 第三层计算（输出层通常不使用激活函数）
        x = self.fc3(x)
        return x

# 创建模型实例
input_size = 1    # 输入特征数（x坐标）
hidden_size1 = 64 # 第一层隐藏层节点数
hidden_size2 = 32 # 第二层隐藏层节点数
output_size = 1   # 输出特征数（y值）

model = MultiLayerNet(input_size, hidden_size1, hidden_size2, output_size)
print(f"多层神经网络模型创建完成:")
print(f"网络结构: {input_size} -> {hidden_size1} -> {hidden_size2} -> {output_size}")
print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss() # 回归任务使用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化器，学习率更大一些

# 4. 训练多层神经网络
num_epochs = 1000
loss_history = [] # 记录loss历史用于可视化

print("开始训练多层神经网络拟合sin函数...")
for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式
    
    # 前向传播：通过神经网络计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 记录损失值
    loss_history.append(loss.item())
    
    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 训练完成后的评估
print("\n训练完成！")
print(f"最终Loss: {loss_history[-1]:.6f}")
print(f"Loss改善: {loss_history[0] - loss_history[-1]:.6f}")
print("---" * 10)

# 6. 可视化结果
# 生成更多点用于绘制平滑的拟合曲线
X_test_numpy = np.linspace(0, 2*np.pi, 300).reshape(-1, 1)
X_test = torch.from_numpy(X_test_numpy).float()

# 使用训练好的模型进行预测
model.eval() # 设置模型为评估模式
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_train = model(X)

# 转换为numpy数组用于绘图
y_pred_test_np = y_pred_test.numpy()
y_pred_train_np = y_pred_train.numpy()

# 创建图形
plt.figure(figsize=(15, 5))

# 子图1：显示训练数据和拟合结果
plt.subplot(1, 3, 1)
plt.scatter(X_numpy, y_numpy, label='训练数据', color='blue', alpha=0.6, s=20)
plt.plot(X_test_numpy, y_pred_test_np, label='神经网络拟合', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linewidth=2, linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sin函数拟合结果')
plt.legend()
plt.grid(True)

# 子图2：显示loss变化曲线
plt.subplot(1, 3, 2)
plt.plot(loss_history, color='purple', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练Loss变化')
plt.grid(True)

# 子图3：显示残差（预测值与真实值的差异）
plt.subplot(1, 3, 3)
residuals = y_pred_train_np - np.sin(X_numpy)
plt.scatter(X_numpy, residuals, color='red', alpha=0.6, s=20)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('X')
plt.ylabel('残差')
plt.title('拟合残差')
plt.grid(True)

plt.tight_layout()
plt.show()
