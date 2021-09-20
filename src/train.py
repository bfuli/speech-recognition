# encoding=utf-8
# Author:fuli
# Date:2021/9/19
import torch
from torch.utils.data import DataLoader
from torch import nn

from dataset import AudioDataSet
from net import LinearNet, ConvNet
import matplotlib.pyplot as plt


def one_train(data_loader):
    total_size = len(data_loader.dataset)
    model.train()
    for batch, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)

        pred = model(data)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total_size:>5d}]")


def one_test(data_loader):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return [100 * correct]


# 标签文件位置
annotation_file = r"res\desc.txt"
# 图像文件的根目录
audio_dir = r"C:\Users\fuli\Desktop\ABCD\录音_转换后"

batch_size = 4

# 标记数据集中用于训练样本的比例，用于测试的样本就是(1-train_ratio)
train_ratio = 0.8

train_data = AudioDataSet(annotation_file, audio_dir, is_train=True, train_ratio=train_ratio)
test_data = AudioDataSet(annotation_file, audio_dir, is_train=False, train_ratio=train_ratio)

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, True)

device = "cuda" if torch.cuda.is_available() else "cpu"

epoch = 20
model = ConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# x:表示训练轮次；y:表示每一轮的测试精确度
# 用于绘制图线，直观展示精确度随训练次数的变化趋势
x, y = [], []

for i in range(epoch):
    print(f"第{i + 1}轮训练.............")
    one_train(train_loader)  # 进行一轮训练

    x += [i + 1]
    y += one_test(test_loader)  # 进行一次测试，并返回该次测试的精确度

# 保存训练好的模型，model_conv1：表示卷积网络的第一个版本
torch.save(model, "saves/model_conv1.pth")

# 绘制准确度曲线
plt.title("Accuracy:lr=" + str(learning_rate))
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.plot(x, y)
plt.show()
