# 导入pandas库，用于数据处理
import pandas as pd
# 导入PyTorch库
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的优化器模块
import torch.optim as optim
# 从PyTorch数据工具中导入数据集和数据加载器类
from torch.utils.data import Dataset, DataLoader

# 读取数据集文件，制表符分隔，无表头
dataset = pd.read_csv("../../dataset.csv", sep="\t", header=None)
# 提取第一列作为文本列表
texts = dataset[0].tolist()
# 提取第二列作为标签列表
string_labels = dataset[1].tolist()

# 创建标签到索引的映射字典
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数字标签
numerical_labels = [label_to_index[label] for label in string_labels]

# 初始化字符到索引的映射字典，包含填充字符
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符到索引的完整映射
for text in texts:
    # 遍历文本中的每个字符
    for char in text:
        # 如果字符不在字典中，则添加
        if char not in char_to_index:
            # 分配新索引值为当前字典长度
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射字典
index_to_char = {i: char for char, i in char_to_index.items()}
# 计算词汇表大小
vocab_size = len(char_to_index)

# 设置最大文本长度
max_len = 40


# 定义字符词袋数据集类
class CharBoWDataset(Dataset):
    # 初始化函数
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        # 存储文本列表
        self.texts = texts
        # 将标签转换为PyTorch张量，类型为长整型
        self.labels = torch.tensor(labels, dtype=torch.long)
        # 存储字符到索引的映射
        self.char_to_index = char_to_index
        # 存储最大长度
        self.max_len = max_len
        # 存储词汇表大小
        self.vocab_size = vocab_size
        # 创建词袋向量
        self.bow_vectors = self._create_bow_vectors()

    # 创建词袋向量的内部方法
    def _create_bow_vectors(self):
        # 初始化分词后的文本列表
        tokenized_texts = []
        # 遍历每个文本
        for text in self.texts:
            # 将字符转换为索引，截取前max_len个字符
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 填充到最大长度
            tokenized += [0] * (self.max_len - len(tokenized))
            # 添加到列表
            tokenized_texts.append(tokenized)

        # 初始化词袋向量列表
        bow_vectors = []
        # 遍历每个分词后的文本
        for text_indices in tokenized_texts:
            # 创建全零向量，长度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            # 遍历每个索引
            for index in text_indices:
                # 如果不是填充字符
                if index != 0:
                    # 在对应位置计数加1
                    bow_vector[index] += 1
            # 添加到列表
            bow_vectors.append(bow_vector)
        # 将所有向量堆叠成张量并返回
        return torch.stack(bow_vectors)

    # 返回数据集长度的方法
    def __len__(self):
        # 返回文本数量
        return len(self.texts)

    # 获取单个样本的方法
    def __getitem__(self, idx):
        # 返回指定索引的词袋向量和标签
        return self.bow_vectors[idx], self.labels[idx]


# 定义简单分类器类
class SimpleClassifier(nn.Module):
    # 初始化函数，新增层数和各层大小参数
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, layer_sizes=None,dropout=0.2):
        # 调用父类初始化
        super(SimpleClassifier, self).__init__()

        # 如果未指定各层大小，则根据层数自动设置
        if layer_sizes is None:
            # 初始化层大小列表
            layer_sizes = []
            for i in range(num_layers - 1,-1,-1):
                layer_sizes.append(pow(hidden_dim,i))

        # 初始化层列表
        layers = []
        # 设置前一层大小为输入维度
        prev_size = input_dim

        # 遍历每个隐藏层
        for layer_size in layer_sizes:
            # 添加线性层
            layers.append(nn.Linear(prev_size, layer_size))
            # 2. 批归一化
            layers.append(nn.BatchNorm1d(layer_size))

            # 添加ReLU激活函数
            layers.append(nn.ReLU())
            # 更新前一层大小
            prev_size = layer_size

        # 添加输出层
        layers.append(nn.Linear(prev_size, output_dim))


        # 将层列表转换为Sequential模块
        self.network = nn.Sequential(*layers)
        # 在ReLU后添加Dropout,增加正则防止过拟合
        layers.append(nn.Dropout(dropout))


    # 前向传播函数
    def forward(self, x):
        # 通过整个网络
        return self.network(x)


# 创建数据集实例
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
# 创建数据加载器
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 设置隐藏层基准维度
hidden_dim = 128
# 计算输出维度
output_dim = len(label_to_index)

# 设置模型层数为4
num_layers = 4
# 设置每层节点个数：[512, 256, 128]
layer_sizes = [512, 256, 128]
# 创建模型实例
model = SimpleClassifier(vocab_size, hidden_dim, output_dim, num_layers, layer_sizes)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
#增加学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 设置训练轮数
num_epochs = 10
# 开始训练循环
for epoch in range(num_epochs):
    # 设置模型为训练模式
    model.train()
    # 初始化累计损失
    running_loss = 0.0
    # 遍历数据加载器中的批次
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累加损失
        running_loss += loss.item()
        # 每50个批次打印一次
        if idx % 50 == 0:
            # 打印批次信息和当前损失
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本转换为索引序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到最大长度
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    # 遍历索引序列
    for index in tokenized:
        # 如果不是填充字符
        if index != 0:
            # 在对应位置计数加1
            bow_vector[index] += 1

    # 增加批次维度
    bow_vector = bow_vector.unsqueeze(0)

    # 设置模型为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 前向传播
        output = model(bow_vector)

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    # 提取索引值
    predicted_index = predicted_index.item()
    # 将索引转换为标签
    predicted_label = index_to_label[predicted_index]

    # 返回预测标签
    return predicted_label


# 创建索引到标签的映射字典
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试文本1
new_text = "帮我导航到北京"
# 进行预测
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
# 打印结果
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 测试文本2
new_text_2 = "查询明天北京的天气"
# 进行预测
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
# 打印结果
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")