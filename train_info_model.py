import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from draw_pic import draw
import csv

##预设变量
out_pic_path = 'output_v4.png'
out_csv_path = 'modelv4_test.csv'
out_model_name = 'info.pt'
data_path = './data_v3.txt'

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.sentences, self.labels = self.load_data(data_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        sentences = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                print(f"具体内容: {line}")
                sentence, label = line.strip().split('&')
                sentences.append(sentence)
                labels.append(int(label))
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # 使用BERT的tokenizer对句子进行编码
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }

#自定义损失函数
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


#测试结果持久化
def save(data_path,save_path,model, tokenizer):
    data = []
    with open(save_path, 'w', newline='',encoding="utf-8") as save_file:    
        writer = csv.writer(save_file)           
        with open(data_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                sentence, label = line.strip().split('&')
                predicted_label = predict_label(sentence, model, tokenizer)
                data = [sentence,predicted_label,label]
                writer.writerow(data)
                print("写入中.....")
        print("测试结果写入完成")

# 设置训练参数
model_name = r'C:\Users\Administrator\Desktop\physical_examination_analysis\chinese-roberta-wwm-ext-lagre'
batch_size = 32
label_cnt = 4               #修改标签
learning_rate = 5e-6
num_epochs = 27

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.num_hidden_layers = 2
config.num_labels = label_cnt   # 将 num_labels 添加到 config 中
model = BertForSequenceClassification.from_pretrained(model_name, config=config)
# model.classifier.add_module('dropout', torch.nn.Dropout(p=0.5))
# model.classifier.add_module('fc1', torch.nn.Linear(768, 256))  # 在原有的分类器上增加一个全连接层
# model.classifier.add_module('fc2', torch.nn.Linear(256, 128)) # 再加一层
# model.classifier.add_module('fc3', torch.nn.Linear(128, 7))

####下面是训练环境

# 加载数据集
dataset = CustomDataset(data_path, tokenizer)


# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(           #这里是随机拆分的，随机种子得到源码里找
    dataset, [train_size, valid_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# 定义优化器和损失函数
# weight_decay为L2正则化 防止过拟合
# scheduler 配置学习率衰减
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=18, gamma=0.1)
# loss_fn = torch.nn.CrossEntropyLoss()                   #这里采用的是交叉熵损失函数
loss_fn = FocalLoss(gamma=2, weight=None)                 ##这里采用的是focal损失函数

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = torch.DataParallel(model, device_ids=[0, 1, 2])

model.to(device)
print(f"该实验采用的设备为：{device}")
 
acc_list,loss_list= [],[]
## 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in train_loader:
        print(f"第{epoch}次训练中")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(outputs.logits, dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)
    # 在验证集上评估模型
    model.eval()
    valid_loss = 0
    correct_valid_predictions = 0
    total_valid_predictions = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)

            valid_loss += loss_fn(logits, labels).item()
            correct_valid_predictions += (predicted_labels == labels).sum().item()
            total_valid_predictions += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_valid_loss = valid_loss / len(valid_loader)
    accuracy = correct_predictions / total_predictions
    valid_accuracy = correct_valid_predictions / total_valid_predictions
    scheduler.step()

    acc_list.append(accuracy)
    loss_list.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Valid Loss: {epoch_valid_loss:.4f} - Accuracy: {accuracy:.4f} - Valid Accuracy: {valid_accuracy:.4f}')



# 计算整体准确率
v_loader = DataLoader(dataset)
model.eval()
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
        for batch in v_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

valid_accuracy = correct_predictions / total_predictions

print(valid_accuracy)
 
# 保存模型
torch.save(model.state_dict(), out_model_name)

# 绘制训练过程
print(f"准确率{acc_list}\n损失情况{loss_list}")
draw(acc_list,loss_list,"文本分类器训练过程",'外迭代次数',"对应曲线数值",out_pic_path)

####下面是测试环境

def predict_label(sentence, model, tokenizer):
    # 对输入句子进行编码
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()

    # 在模型中进行前向传播
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits

    # 获取预测的标签
    _, predicted_label = torch.max(logits, dim=1)

    return predicted_label.item()


# # 加载训练好的模型参数
model.load_state_dict(torch.load(out_model_name), strict=False)

# 设置模型为评估模式
model.eval()

# 测试栗子
# 0是垃圾信息，1是受检者姓名，2是体检机构，3是体检日期
save(data_path,out_csv_path, model, tokenizer)

