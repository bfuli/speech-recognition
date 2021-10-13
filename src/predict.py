# encoding=utf-8
# Author:fuli
# Date:2021/9/21
import torch
from dataset import AudioDataSet
from torch.utils.data import DataLoader
import os
from torch import nn


def test_audio(model):
    """
    使用训练之后保存的模型，用自己的语音文件进行预测，查看模型效果
    注意：在输入测试文件名时，文件所在文件夹为：src/res/转换后/
    :param model:
    :return:
    """
    classes = ["A", "B", "C", "D"]
    while True:
        try:
            audio_file = input("input audio file:")
            data = AudioDataSet.audio2data(os.path.join("res/转换后/wav", audio_file)).to(device)
            data = data.unsqueeze(0)
            model.eval()
            pred = model(data)
            print("Prediction:" + classes[pred.argmax(1).item()])
        except:
            print("open error, try again!")
            continue


def get_dataset():
    annotation_file = r"C:\Users\fuli\Desktop\wav_data\annotation_file_bfl.txt"
    audio_dir = r"C:\Users\fuli\Desktop\wav_data"

    with open(annotation_file, encoding="utf-8") as f:
        lines = f.readlines()
    f.close()

    dataset = AudioDataSet(lines, audio_dir, is_train=False, train_ratio=0, is_silence=False)
    return dataset


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


def test_audios(model, data_loader):
    # 测试一批的数据
    one_test(data_loader)
    category_Accuracy(model, data_loader)


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "model_conv8_data_shuffle_per_lr=1e-05.pth"
model_path = "saves/" + model_name
model = torch.load(model_path)
loss_fn = nn.CrossEntropyLoss()

test_dataloader = DataLoader(get_dataset(), batch_size=4, shuffle=True)

test_audios(model, test_dataloader)

# test_audio(model)
