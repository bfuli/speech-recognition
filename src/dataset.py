# encoding=utf-8
# Author:fuli
# Date:2021/9/19
from torch.utils.data import Dataset, DataLoader
import librosa
import torch
import os
import numpy as np


class AudioDataSet(Dataset):
    def __init__(self, annotation_file, audio_dir, is_train=False, train_ratio=0.9):
        """
        :param annotation_file: 带标注的文件地址，每一行包含(标签，语音文件名)，并以“,”分割
        :param audio_dir: 语音文件根目录
        :param is_train: 训练or测试
        :param train_ratio: 训练数据集占总数据集的比例
        """
        with open(annotation_file, encoding="utf-8") as f:
            lines = f.readlines()

        labels = [line.strip("\n").split(",")[0] for line in lines]
        audio_names = [line.strip("\n").split(",")[1] for line in lines]

        tag = int(len(labels) * train_ratio)
        if is_train:
            self.labels = labels[:tag]
            self.audio_names = audio_names[:tag]
        else:
            self.labels = labels[tag:]
            self.audio_names = audio_names[tag:]
        self.audio_dir = audio_dir
        self.classes = ["A", "B", "C", "D"]

    def __getitem__(self, index):
        """
        :param index:
        :return: 返回：标签的下标， 二维tensor
        """
        filePath = os.path.join(self.audio_dir, self.audio_names[index])
        data = audio2data(filePath)

        return data, self.classes.index(self.labels[index])

    def __len__(self):
        return len(self.labels)


def audio2data(filePath):
    y, sr = librosa.load(filePath, sr=None)
    # 提取Log-MelSpectrogram特征
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    # resampled = np.mean(logmelspec, axis=0)
    resampled = librosa.resample(y=logmelspec, orig_sr=sr, target_sr=np.floor(50 * sr / len(logmelspec[1])),
                                 fix=True)
    return torch.tensor(resampled)
