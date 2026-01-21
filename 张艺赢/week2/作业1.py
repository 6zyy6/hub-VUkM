import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class WideClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WideClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)  # 更宽的隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)

# 定义不同的模型配置
model_configs = [
    {"name": "Simple-64", "model": SimpleClassifier(vocab_size, 64, output_dim), "hidden_dim": 64},
    {"name": "Simple-128", "model": SimpleClassifier(vocab_size, 128, output_dim), "hidden_dim": 128},
    {"name": "Simple-256", "model": SimpleClassifier(vocab_size, 256, output_dim), "hidden_dim": 256},
    {"name": "Deep-64-32", "model": DeepClassifier(vocab_size, 64, 32, output_dim), "hidden_dim1": 64, "hidden_dim2": 32},
    {"name": "Deep-128-64", "model": DeepClassifier(vocab_size, 128, 64, output_dim), "hidden_dim1": 128, "hidden_dim2": 64},
    {"name": "Wide-256", "model": WideClassifier(vocab_size, 128, output_dim), "hidden_dim": 256}
]

# 训练不同配置的模型并记录loss
results = {}
num_epochs = 10

for config in model_configs:
    print(f"\n{'='*50}")
    print(f"训练模型: {config['name']}")
    print(f"{'='*50}")
    
    model = config["model"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_losses = []
        
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_losses.append(loss.item())
            
            if idx % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {idx}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], 平均Loss: {epoch_loss:.4f}")
    
    results[config["name"]] = epoch_losses


# 对比分析不同模型的loss变化
print(f"\n{'='*60}")
print("不同模型配置的Loss对比分析")
print(f"{'='*60}")

# 打印每个模型的最终loss
print("\n模型最终Loss对比:")
for model_name, losses in results.items():
    final_loss = losses[-1]
    initial_loss = losses[0]
    improvement = initial_loss - final_loss
    print(f"{model_name:12s}: 初始Loss={initial_loss:.4f}, 最终Loss={final_loss:.4f}, 改善={improvement:.4f}")

# 找出loss最低的模型
best_model = min(results.keys(), key=lambda x: results[x][-1])
print(f"\n最佳模型: {best_model} (最终Loss: {results[best_model][-1]:.4f})")

# 分析模型复杂度与loss的关系
print(f"\n模型复杂度分析:")
print("Simple-64:   2层，隐藏层64节点")
print("Simple-128:  2层，隐藏层128节点") 
print("Simple-256:  2层，隐藏层256节点")
print("Deep-64-32:  3层，隐藏层64->32节点")
print("Deep-128-64: 3层，隐藏层128->64节点")
print("Wide-256:    2层，隐藏层256节点")

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
