# encoding=utf-8
# Author:fuli
# Date:2021/9/21
import torch
from dataset import audio2data
import os


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
            data = audio2data(os.path.join("res/转换后", audio_file)).to(device)
            data = data.unsqueeze(0)
            model.eval()
            pred = model(data)
            print("Prediction:" + classes[pred.argmax(1).item()])
        except:
            print("open error, try again!")
            continue


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "saves/model_conv1_2_lr=1e-06.pth"
model = torch.load(model_path)
test_audio(model)
