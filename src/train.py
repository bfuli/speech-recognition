# encoding=utf-8
# Author:fuli
# Date:2021/9/19
import torch
from torch.utils.data import DataLoader
from torch import nn

from dataset import AudioDataSet
from net import LinearNet, ConvNet
import matplotlib.pyplot as plt
import time


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

        if batch % 20 == 0:
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


def category_Accuracy(model, data_loader):
    """分别计算每个分类的准确度"""
    classes = ["A", "B", "C", "D"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for datas, labels in data_loader:
            datas = datas.to(device)
            labels = labels.to(device)

            outputs = model(datas)
            preds = outputs.argmax(1)
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))


# 标签文件位置
annotation_file = r"C:\Users\fuli\Desktop\desc.txt"
# 图像文件的根目录
audio_dir = r"C:\Users\fuli\Desktop\录音_转换后"

batch_size = 8

# 标记数据集中用于训练样本的比例，用于测试的样本就是(1-train_ratio)
train_ratio = 0.8

# 获取训练、测试数据集
train_data, test_data = AudioDataSet.get_dataset(annotation_file, audio_dir, train_ratio, is_shuffle=True)

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, True)

device = "cuda" if torch.cuda.is_available() else "cpu"

epoch = 100

model = ConvNet().to(device)
# model_name = "model_conv8_data_shuffle_lr=1e-05.pth"
# model_path = "saves/" + model_name
# model = torch.load(model_path)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# x:表示训练轮次；y:表示每一轮的测试精确度
# 用于绘制图线，直观展示精确度随训练次数的变化趋势
x, y = [], []

# 获取训练开始时间
start = time.time()
for i in range(epoch):
    print(f"第{i + 1}轮训练.............")
    one_train(train_loader)  # 进行一轮训练

    x += [i + 1]
    y += one_test(test_loader)  # 进行一次测试，并返回该次测试的精确度

# 保存训练好的模型，
# model_convi：表示卷积网络的第i个版本,lr：当前学习速率；
# data_shuffle：表示当前使用的数据集是打乱之后的;
# classes：表示分类是["B", "A", "C", "D"] ;classes2: 表示分类是["A", "C", "B", "D"]
# per：表示加入了自己录的语音文件进行训练
model_name = "model_conv8_data_shuffle_per"
torch.save(model, "saves/" + model_name + "_lr=" + str(learning_rate) + ".pth")

# 获取训练结束时间，并计算总耗时，单位：分钟
end = time.time()
print(f"总耗时：{(end - start) / 60:>0.1f} min")

# 测试每个分类的精确度
category_Accuracy(model, test_loader)

# 绘制准确度曲线
plt.title(model_name + ":lr=" + str(learning_rate))
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.plot(x, y)

# 每隔5个点，显示一个点的数值
for a, b in zip(x, y):
    if a % 5 == 0 or b > 90:
        plt.text(a, b, "%.1f" % b, fontdict={"fontsize": 8})
plt.show()
